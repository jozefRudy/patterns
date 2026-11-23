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
      channel = "stable";
      enable = true;
      components = ["rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" "rust-src"];
    };
  };
}
