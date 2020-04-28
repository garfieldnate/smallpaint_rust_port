mod painterly;
// use painterly;
// use crate::painterly;
use clap::{App, Arg, SubCommand};
use std::str::FromStr;

fn main() {
    let matches = App::new("Smallpaint Rust Port")
        .version("0.01")
        .author("Nathan Glenn")
        .about("Experiment with global illumination")
        .subcommand(
            SubCommand::with_name("painterly")
                .about("Exploits a sampling bug to create a paint-like effect")
                .arg(
                    Arg::with_name("refractive_index")
                        .short("r")
                        .takes_value(true)
                        .default_value("1.5")
                        .help("refractive index used to trace all transparent materials (glass=1.5, diamond=2.4)"),
                )
                .arg(
                    Arg::with_name("samples_per_pixel")
                        .short("s")
                        .takes_value(true)
                        .default_value("50")
                        .help("Number of samples per pixel"),
                )
                .arg(
                    Arg::with_name("width")
                        .short("w")
                        .takes_value(true)
                        .default_value("512")
                        .help("Width of output canvas"),
                )
                .arg(
                    Arg::with_name("height")
                        .short("h")
                        .takes_value(true)
                        .default_value("512")
                        .help("Height of output canvas"),
                ),
        )
        .get_matches();

    if let Some(matches) = matches.subcommand_matches("painterly") {
        let params = painterly::Params {
            refractive_index: f64::from_str(matches.value_of("refractive_index").unwrap()).unwrap(),
            samples_per_pixel: u64::from_str(matches.value_of("samples_per_pixel").unwrap())
                .unwrap(),
            width: usize::from_str(matches.value_of("width").unwrap()).unwrap(),
            height: usize::from_str(matches.value_of("height").unwrap()).unwrap(),
        };
        painterly::run(params);
    }
}
