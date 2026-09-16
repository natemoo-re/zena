use anyhow::Result;
use std::io::Error as IoError;
use std::io::ErrorKind;
use wasmtime::*;

use crate::MyState;

const WASM_PAGE: usize = 65536;
const TRANSFER_BUDGET: usize = 1024 * 1024 + 8192 * 116;

pub struct ClayTerm {
    pub instance: Instance,
    pub memory: Memory,
    pub state_ptr: i32,
    pub ops_buf: i32,
    pub input_ptr: i32,
}

fn io_err(msg: &'static str) -> IoError {
    IoError::new(ErrorKind::Other, msg)
}

/// Load clayterm.wasm, wire its imports, and instantiate it into `store`.
pub fn load_clayterm(
    engine: &Engine,
    linker: &mut Linker<MyState>,
    store: &mut Store<MyState>,
    path: &str,
    width: i32,
    height: i32,
) -> Result<ClayTerm> {
    let clay_bytes = std::fs::read(path)?;
    let clay_module = Module::new(engine, &clay_bytes)?;

    let clay_mem = Memory::new(&mut *store, MemoryType::new(2, None))?;
    linker.define(&mut *store, "env", "memory", clay_mem)?;

    // clay.measureTextFunction — called by clay during layout to measure text width.
    let mt_ty = FuncType::new(
        engine,
        [ValType::I32, ValType::I32, ValType::I32, ValType::I32],
        [],
    );
    linker.func_new("clay", "measureTextFunction", mt_ty, |mut caller, params, _| {
        let ret = params[0].unwrap_i32();
        let text = params[1].unwrap_i32();
        let Some(clay) = caller.data().clayterm.as_ref().map(|c| c.instance) else {
            return Ok(());
        };
        if let Some(measure) = clay.get_func(&mut caller, "measure") {
            measure.call(&mut caller, &[Val::I32(ret), Val::I32(text)], &mut [])?;
        }
        Ok(())
    })?;

    // clay.queryScrollOffsetFunction — always return zero scroll offset.
    let qs_ty = FuncType::new(engine, [ValType::I32, ValType::I32, ValType::I32], []);
    linker.func_new("clay", "queryScrollOffsetFunction", qs_ty, |mut caller, params, _| {
        let ret = params[0].unwrap_i32() as usize;
        let Some(mem) = caller.data().clayterm.as_ref().map(|c| c.memory) else {
            return Ok(());
        };
        mem.data_mut(&mut caller)[ret..ret + 4].copy_from_slice(&0f32.to_le_bytes());
        mem.data_mut(&mut caller)[ret + 4..ret + 8].copy_from_slice(&0f32.to_le_bytes());
        Ok(())
    })?;

    let clay_instance = linker.instantiate(&mut *store, &clay_module)?;

    let heap_base = clay_instance
        .get_global(&mut *store, "__heap_base")
        .and_then(|g| match g.get(&mut *store) {
            Val::I32(v) => Some(v),
            _ => None,
        })
        .unwrap_or(0);

    let state_size = clay_instance
        .get_typed_func::<(i32, i32), i32>(&mut *store, "clayterm_size")
        .or_else(|_| clay_instance.get_typed_func::<(i32, i32), i32>(&mut *store, "tty_size"))?
        .call(&mut *store, (width, height))?;

    let needed = heap_base as usize + state_size as usize + TRANSFER_BUDGET;
    let pages = (needed + WASM_PAGE - 1) / WASM_PAGE;
    let current = clay_mem.size(&mut *store) as usize;
    if pages > current {
        clay_mem.grow(&mut *store, (pages - current) as u64)?;
    }

    let state_ptr = clay_instance
        .get_typed_func::<(i32, i32, i32), i32>(&mut *store, "init")?
        .call(&mut *store, (heap_base, width, height))?;

    let ops_buf = (heap_base + state_size + 3) & !3;

    let input_size = clay_instance
        .get_typed_func::<(), i32>(&mut *store, "input_size")?
        .call(&mut *store, ())?;

    let input_ptr = (ops_buf + TRANSFER_BUDGET as i32 + 3) & !3;
    let input_end = input_ptr as usize + input_size as usize;
    if input_end > clay_mem.data_size(&mut *store) {
        let extra = (input_end - clay_mem.data_size(&mut *store) + WASM_PAGE - 1) / WASM_PAGE;
        clay_mem.grow(&mut *store, extra as u64)?;
    }

    clay_instance
        .get_typed_func::<(i32, i32), ()>(&mut *store, "input_init")?
        .call(&mut *store, (input_ptr, 0))?;

    Ok(ClayTerm { instance: clay_instance, memory: clay_mem, state_ptr, ops_buf, input_ptr })
}

/// Register all `env.clayterm_*` and `env.tty_*` host imports on `linker`.
pub fn add_tty_imports(engine: &Engine, linker: &mut Linker<MyState>) -> Result<()> {
    // ── clayterm_reduce ───────────────────────────────────────────────────────
    linker.func_new(
        "env",
        "clayterm_reduce",
        FuncType::new(
            engine,
            [ValType::I32, ValType::I32, ValType::I32, ValType::I32, ValType::F64],
            [],
        ),
        |mut caller, params, _| {
            let ops_ptr = params[0].unwrap_i32() as usize;
            let ops_len = params[1].unwrap_i32() as usize;
            let mode    = params[2].unwrap_i32();
            let row     = params[3].unwrap_i32();
            let dt      = params[4].unwrap_f64();

            let (clay_inst, clay_mem, state_ptr, clay_ops_buf) = {
                let ct = caller.data().clayterm.as_ref().ok_or(io_err("clayterm not initialized"))?;
                (ct.instance, ct.memory, ct.state_ptr, ct.ops_buf)
            };

            let zena_mem = caller.get_export("memory")
                .and_then(|e| e.into_memory())
                .ok_or(io_err("no Zena memory export"))?;

            let ops_bytes: Vec<u8> =
                zena_mem.data(&caller)[ops_ptr..ops_ptr + ops_len].to_vec();
            clay_mem.data_mut(&mut caller)
                [clay_ops_buf as usize..clay_ops_buf as usize + ops_len]
                .copy_from_slice(&ops_bytes);

            clay_inst
                .get_typed_func::<(i32, i32, i32, i32, i32, f64), ()>(&mut caller, "reduce")?
                .call(&mut caller, (state_ptr, clay_ops_buf, ops_len as i32, mode, row, dt))?;
            Ok(())
        },
    )?;

    // ── clayterm_flush ────────────────────────────────────────────────────────
    linker.func_new("env", "clayterm_flush", FuncType::new(engine, [], []), |mut caller, _, _| {
        let (clay_inst, clay_mem, state_ptr) = {
            let ct = caller.data().clayterm.as_ref().ok_or(io_err("clayterm not initialized"))?;
            (ct.instance, ct.memory, ct.state_ptr)
        };

        let ptr = clay_inst
            .get_typed_func::<i32, i32>(&mut caller, "output")?
            .call(&mut caller, state_ptr)? as usize;
        let len = clay_inst
            .get_typed_func::<i32, i32>(&mut caller, "length")?
            .call(&mut caller, state_ptr)? as usize;

        if len > 0 {
            let bytes: Vec<u8> = clay_mem.data(&caller)[ptr..ptr + len].to_vec();
            use std::io::Write;
            std::io::stdout().write_all(&bytes)?;
            std::io::stdout().flush()?;
        }
        Ok(())
    })?;

    // ── clayterm_animating ────────────────────────────────────────────────────
    linker.func_new(
        "env",
        "clayterm_animating",
        FuncType::new(engine, [], [ValType::I32]),
        |mut caller, _, results| {
            let (clay_inst, state_ptr) = {
                let ct = caller.data().clayterm.as_ref().ok_or(io_err("clayterm not initialized"))?;
                (ct.instance, ct.state_ptr)
            };
            let v = clay_inst
                .get_typed_func::<i32, i32>(&mut caller, "animating")?
                .call(&mut caller, state_ptr)?;
            results[0] = Val::I32(v);
            Ok(())
        },
    )?;

    // ── clayterm_input_scan ───────────────────────────────────────────────────
    linker.func_new(
        "env",
        "clayterm_input_scan",
        FuncType::new(engine, [ValType::I32, ValType::I32], []),
        |mut caller, params, _| {
            let buf_ptr = params[0].unwrap_i32() as usize;
            let buf_len = params[1].unwrap_i32() as usize;

            let (clay_inst, clay_mem, staging, input_ptr) = {
                let ct = caller.data().clayterm.as_ref().ok_or(io_err("clayterm not initialized"))?;
                (ct.instance, ct.memory, ct.ops_buf as usize, ct.input_ptr)
            };

            let zena_mem = caller.get_export("memory")
                .and_then(|e| e.into_memory())
                .ok_or(io_err("no Zena memory export"))?;

            let bytes: Vec<u8> = zena_mem.data(&caller)[buf_ptr..buf_ptr + buf_len].to_vec();
            clay_mem.data_mut(&mut caller)[staging..staging + buf_len]
                .copy_from_slice(&bytes);

            clay_inst
                .get_typed_func::<(i32, i32, i32), ()>(&mut caller, "input_scan")?
                .call(&mut caller, (input_ptr, staging as i32, buf_len as i32))?;
            Ok(())
        },
    )?;

    // ── clayterm_input_count ──────────────────────────────────────────────────
    linker.func_new(
        "env",
        "clayterm_input_count",
        FuncType::new(engine, [], [ValType::I32]),
        |mut caller, _, results| {
            let (clay_inst, input_ptr) = {
                let ct = caller.data().clayterm.as_ref().ok_or(io_err("clayterm not initialized"))?;
                (ct.instance, ct.input_ptr)
            };
            let n = clay_inst
                .get_typed_func::<i32, i32>(&mut caller, "input_count")?
                .call(&mut caller, input_ptr)?;
            results[0] = Val::I32(n);
            Ok(())
        },
    )?;

    // ── clayterm_input_event ──────────────────────────────────────────────────
    linker.func_new(
        "env",
        "clayterm_input_event",
        FuncType::new(engine, [ValType::I32, ValType::I32], []),
        |mut caller, params, _| {
            let index   = params[0].unwrap_i32();
            let out_ptr = params[1].unwrap_i32() as usize;

            let (clay_inst, clay_mem, input_ptr) = {
                let ct = caller.data().clayterm.as_ref().ok_or(io_err("clayterm not initialized"))?;
                (ct.instance, ct.memory, ct.input_ptr)
            };

            let event_ptr = clay_inst
                .get_typed_func::<(i32, i32), i32>(&mut caller, "input_event")?
                .call(&mut caller, (input_ptr, index))? as usize;

            const EVENT_SIZE: usize = 64;
            let event_bytes: Vec<u8> =
                clay_mem.data(&caller)[event_ptr..event_ptr + EVENT_SIZE].to_vec();

            let zena_mem = caller.get_export("memory")
                .and_then(|e| e.into_memory())
                .ok_or(io_err("no Zena memory export"))?;
            zena_mem.data_mut(&mut caller)[out_ptr..out_ptr + EVENT_SIZE]
                .copy_from_slice(&event_bytes);
            Ok(())
        },
    )?;

    // ── clayterm_input_delay ──────────────────────────────────────────────────
    linker.func_new(
        "env",
        "clayterm_input_delay",
        FuncType::new(engine, [], [ValType::F64]),
        |mut caller, _, results| {
            let (clay_inst, input_ptr) = {
                let ct = caller.data().clayterm.as_ref().ok_or(io_err("clayterm not initialized"))?;
                (ct.instance, ct.input_ptr)
            };
            let d = clay_inst
                .get_typed_func::<i32, f64>(&mut caller, "input_delay")?
                .call(&mut caller, input_ptr)?;
            results[0] = Val::F64(d.to_bits());
            Ok(())
        },
    )?;

    // ── tty_enable_raw_mode ───────────────────────────────────────────────────
    linker.func_new("env", "tty_enable_raw_mode", FuncType::new(engine, [], []), |_, _, _| {
        enable_raw_mode().map_err(|e| IoError::other(e.to_string()))?;
        Ok(())
    })?;

    // ── tty_disable_raw_mode ──────────────────────────────────────────────────
    linker.func_new("env", "tty_disable_raw_mode", FuncType::new(engine, [], []), |_, _, _| {
        disable_raw_mode().map_err(|e| IoError::other(e.to_string()))?;
        Ok(())
    })?;

    // ── tty_read_stdin ────────────────────────────────────────────────────────
    linker.func_new(
        "env",
        "tty_read_stdin",
        FuncType::new(engine, [ValType::I32, ValType::I32], [ValType::I32]),
        |mut caller, params, results| {
            let buf_ptr = params[0].unwrap_i32() as usize;
            let buf_len = params[1].unwrap_i32() as usize;
            let zena_mem = caller.get_export("memory")
                .and_then(|e| e.into_memory())
                .ok_or(io_err("no Zena memory export"))?;
            use std::io::Read;
            let n = match std::io::stdin()
                .read(&mut zena_mem.data_mut(&mut caller)[buf_ptr..buf_ptr + buf_len])
            {
                Ok(n) => n as i32,
                Err(_) => -1,
            };
            results[0] = Val::I32(n);
            Ok(())
        },
    )?;

    // ── tty_get_cols / tty_get_rows ───────────────────────────────────────────
    linker.func_new(
        "env",
        "tty_get_cols",
        FuncType::new(engine, [], [ValType::I32]),
        |_, _, results| { results[0] = Val::I32(terminal_size().0); Ok(()) },
    )?;
    linker.func_new(
        "env",
        "tty_get_rows",
        FuncType::new(engine, [], [ValType::I32]),
        |_, _, results| { results[0] = Val::I32(terminal_size().1); Ok(()) },
    )?;

    Ok(())
}

// ── Terminal OS primitives ────────────────────────────────────────────────────

pub fn terminal_size_i32() -> (i32, i32) {
    terminal_size()
}

fn terminal_size() -> (i32, i32) {
    #[cfg(unix)]
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) == 0 {
            return (ws.ws_col as i32, ws.ws_row as i32);
        }
    }
    (80, 24)
}

fn enable_raw_mode() -> Result<()> {
    #[cfg(unix)]
    unsafe {
        let mut t: libc::termios = std::mem::zeroed();
        if libc::tcgetattr(libc::STDIN_FILENO, &mut t) != 0 {
            anyhow::bail!("tcgetattr failed");
        }
        libc::cfmakeraw(&mut t);
        if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &t) != 0 {
            anyhow::bail!("tcsetattr failed");
        }
    }
    Ok(())
}

fn disable_raw_mode() -> Result<()> {
    #[cfg(unix)]
    unsafe {
        let mut t: libc::termios = std::mem::zeroed();
        if libc::tcgetattr(libc::STDIN_FILENO, &mut t) != 0 {
            anyhow::bail!("tcgetattr failed");
        }
        t.c_lflag |= libc::ICANON | libc::ECHO;
        if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &t) != 0 {
            anyhow::bail!("tcsetattr failed");
        }
    }
    Ok(())
}
