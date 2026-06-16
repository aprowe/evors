//! Run the benchmark sweep: print a per-problem leaderboard and write CSVs.

use evors_bench::harness::{Summary, default_configs, summarize};
use evors_bench::problems::all;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};

const BUDGET: usize = 20_000;
const SEEDS: u64 = 15;

fn main() {
    create_dir_all("data/bench").unwrap();
    let configs = default_configs();
    let problems = all();
    let mut results: Vec<Summary> = Vec::new();

    for p in &problems {
        println!("\n=== {} (global_min {:.4}) ===", p.name, p.global_min);
        println!(
            "{:<34} {:>10} {:>10} {:>8} {:>10} {:>9}",
            "config", "best", "error", "succ%", "med-evals", "coverage"
        );
        let mut summaries: Vec<Summary> = configs
            .iter()
            .map(|c| summarize(p, c, BUDGET, SEEDS))
            .collect();
        // Leaderboard: best error first, then highest coverage.
        summaries.sort_by(|a, b| {
            a.mean_error
                .partial_cmp(&b.mean_error)
                .unwrap()
                .then(b.mean_coverage.partial_cmp(&a.mean_coverage).unwrap())
        });
        for s in &summaries {
            let evals = s
                .median_evals
                .map(|e| e.to_string())
                .unwrap_or_else(|| "-".into());
            println!(
                "{:<34} {:>10.4} {:>10.4} {:>7.0}% {:>10} {:>9.1}",
                s.config,
                s.mean_best,
                s.mean_error,
                s.success_rate * 100.0,
                evals,
                s.mean_coverage
            );
        }
        write_convergence(p.name, &summaries);
        write_scatter(p.name, &summaries);
        results.extend(summaries);
    }
    write_results(&results);
    println!("\nwrote data/bench/results.csv + per-problem convergence/scatter CSVs");
}

fn write_results(results: &[Summary]) {
    let f = File::create("data/bench/results.csv").unwrap();
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "problem,config,mean_best,mean_error,success_rate,median_evals,mean_coverage"
    )
    .unwrap();
    for s in results {
        let evals = s.median_evals.map(|e| e.to_string()).unwrap_or_default();
        writeln!(
            w,
            "{},\"{}\",{},{},{},{},{}",
            s.problem, s.config, s.mean_best, s.mean_error, s.success_rate, evals, s.mean_coverage
        )
        .unwrap();
    }
}

fn write_convergence(problem: &str, summaries: &[Summary]) {
    let f = File::create(format!("data/bench/{problem}_convergence.csv")).unwrap();
    let mut w = BufWriter::new(f);
    let labels: Vec<String> = summaries
        .iter()
        .map(|s| format!("\"{}\"", s.config))
        .collect();
    writeln!(w, "eval,{}", labels.join(",")).unwrap();
    let len = summaries
        .iter()
        .map(|s| s.mean_convergence.len())
        .min()
        .unwrap_or(0);
    for i in (0..len).step_by(50) {
        let row: Vec<String> = summaries
            .iter()
            .map(|s| s.mean_convergence[i].to_string())
            .collect();
        writeln!(w, "{i},{}", row.join(",")).unwrap();
    }
}

fn write_scatter(problem: &str, summaries: &[Summary]) {
    let f = File::create(format!("data/bench/{problem}_scatter.csv")).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "config,x,y").unwrap();
    for s in summaries {
        for x in &s.sample_pool {
            if x.len() >= 2 {
                writeln!(w, "\"{}\",{},{}", s.config, x[0], x[1]).unwrap();
            }
        }
    }
}
