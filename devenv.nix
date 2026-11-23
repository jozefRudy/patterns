{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: {
  packages = [pkgs.git];

  languages = {
    rust = {
      version = "1.100.0";
      enable = true;
      components = ["rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" "rust-src"];
    };
  };
}
