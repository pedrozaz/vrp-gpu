use crate::validation::{check_matrix, check_move, check_reversal, cost, synthetic};
use std::{
    error::Error,
    fs::{self, OpenOptions},
    hint::black_box,
    io::Write,
    path::PathBuf,
    time::Instant,
};
use vrp_gpu::{
    construct::nearest_neighbor,
    instance::SolomonInstance,
    local_search::{cpu, gpu},
    solution::Route,
};

struct Options {
    output: PathBuf,
    samples: usize,
    warmup: usize,
    solomon: Vec<PathBuf>,
}

fn options(args: impl Iterator<Item = String>) -> Result<Option<Options>, String> {
    let mut args = args.peekable();
    if args.peek().is_some_and(|s| s == "--help") {
        println!(
            "Usage: vrp-gpu-bench --output NEW.csv [--samples 20] [--warmup 3] [--solomon FILE]...\nRun in release mode with --features gpu. Synthetic cases always run; Solomon routes use nearest-neighbor construction outside timing."
        );
        return Ok(None);
    }
    let mut output = None;
    let mut samples = 20;
    let mut warmup = 3;
    let mut solomon = Vec::new();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--output" if output.is_none() => output = Some(PathBuf::from(value)),
            "--samples" => samples = value.parse().map_err(|_| "invalid sample count")?,
            "--warmup" => warmup = value.parse().map_err(|_| "invalid warmup count")?,
            "--solomon" => solomon.push(PathBuf::from(value)),
            _ => return Err(format!("unknown or repeated option: {flag}")),
        }
    }
    if !(2..=10000).contains(&samples) || !(1..=1000).contains(&warmup) {
        return Err("samples must be 2..=10000; warmup must be 1..=1000".into());
    }
    Ok(Some(Options {
        output: output.ok_or("--output is required")?,
        samples,
        warmup,
        solomon,
    }))
}

pub(crate) fn run() -> Result<(), Box<dyn Error>> {
    let Some(options) = options(std::env::args().skip(1))? else {
        return Ok(());
    };
    if cfg!(debug_assertions) {
        return Err("timing requires --release".into());
    }
    // Complete input preparation and correctness validation before recording any timings.
    let mut cases = Vec::new();
    for n in [0, 1, 2, 16, 17, 65, 257, 1024, 2048] {
        let (instance, route) = synthetic(n);
        cases.push((format!("synthetic-{n}"), instance, vec![route]));
    }
    for (index, path) in options.solomon.iter().enumerate() {
        let instance: SolomonInstance = fs::read_to_string(path)?.parse()?;
        let solution = nearest_neighbor(&instance);
        if !solution.is_feasible(&instance) {
            return Err("infeasible initial solution".into());
        }
        cases.push((format!("solomon-{index}"), instance, solution.routes));
    }
    for (label, instance, routes) in &cases {
        instance.validate()?;
        for (index, route) in routes.iter().enumerate() {
            check_matrix(
                route,
                instance,
                &gpu::evaluate_two_opt_deltas(route, instance)?,
            )?;
            let expected = cpu::best_two_opt_move(route, instance);
            check_move(gpu::best_two_opt_move(route, instance)?, expected)?;
            check_reversal(route, instance, expected)?;
            eprintln!("validated {label} route {index}, {} customers", route.len());
        }
    }
    // Buffer the complete successful run so a failed measurement never looks complete.
    let mut csv = String::from(
        "case,route,customers,candidates,sample,order,cpu_ns,gpu_ns,selected_i,selected_j,delta,cost_before_f64,cost_after_f64\n",
    );
    for (label, instance, routes) in &cases {
        for (index, route) in routes.iter().enumerate() {
            measure(&mut csv, label, index, route, instance, &options)?;
        }
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&options.output)?;
    file.write_all(csv.as_bytes())?;
    file.sync_all()?;
    eprintln!(
        "saved {} (all correctness checks passed)",
        options.output.display()
    );
    Ok(())
}

fn measure(
    csv: &mut String,
    label: &str,
    index: usize,
    route: &Route,
    instance: &SolomonInstance,
    options: &Options,
) -> Result<(), Box<dyn Error>> {
    use std::fmt::Write;
    let expected = cpu::best_two_opt_move(route, instance);
    let after = check_reversal(route, instance, expected)?;
    for _ in 0..options.warmup {
        black_box(cpu::best_two_opt_move(
            black_box(route),
            black_box(instance),
        ));
        check_move(
            gpu::best_two_opt_move(black_box(route), black_box(instance))?,
            expected,
        )?;
    }
    for sample in 0..options.samples {
        let cpu_call = || {
            let start = Instant::now();
            let result = black_box(cpu::best_two_opt_move(
                black_box(route),
                black_box(instance),
            ));
            (start.elapsed().as_nanos(), result)
        };
        let gpu_call = || {
            let start = Instant::now();
            let result = black_box(gpu::best_two_opt_move(
                black_box(route),
                black_box(instance),
            ));
            (start.elapsed().as_nanos(), result)
        };
        let (cpu_result, gpu_result) = if sample % 2 == 0 {
            let c = cpu_call();
            (c, gpu_call())
        } else {
            let g = gpu_call();
            (cpu_call(), g)
        };
        check_move(cpu_result.1, expected)?;
        check_move(gpu_result.1?, expected)?;
        let (i, j, delta) = expected.map_or((String::new(), String::new(), String::new()), |m| {
            (m.i.to_string(), m.j.to_string(), m.delta.to_string())
        });
        let n = route.len();
        writeln!(
            csv,
            "{label},{index},{n},{},{sample},{},{},{},{i},{j},{delta},{},{}",
            n * n.saturating_sub(1) / 2,
            if sample % 2 == 0 {
                "cpu-first"
            } else {
                "gpu-first"
            },
            cpu_result.0,
            gpu_result.0,
            cost(route, instance),
            after
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cli_rejects_invalid_sampling_and_unknown_options() {
        for args in [
            vec![],
            vec!["--output"],
            vec!["--output", "x", "--samples", "0"],
            vec!["--output", "x", "--warmup", "0"],
            vec!["--output", "x", "--typo", "2"],
        ] {
            assert!(options(args.into_iter().map(str::to_owned)).is_err());
        }
        let parsed = options(
            [
                "--output",
                "x",
                "--samples",
                "2",
                "--solomon",
                "a",
                "--solomon",
                "b",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap()
        .unwrap();
        assert_eq!(parsed.samples, 2);
        assert_eq!(parsed.solomon.len(), 2);
    }
}
