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
      enable = true;
      channel = "nightly";
    };
  };
  scripts = {
    export-docs.exec = ''
      RUSTDOCFLAGS="-Zunstable-options --output-format=json" cargo doc --workspace
      TARGET_DIR=$(cargo metadata --format-version 1 --no-deps | jq -r .target_directory)
      JSON_DIR=$(mktemp -d)
      for name in $(cargo metadata --format-version 1 | jq -r '. as $m | ([$m.resolve.nodes[] | select(.id as $id | $m.workspace_members | index($id)) | .deps[].pkg] + $m.workspace_members) | unique[] as $pid | $m.packages[] | select(.id == $pid) | .name' | sort -u); do
        json="$TARGET_DIR/doc/''${name//-/_}.json"
        [ -f "$json" ] && ln -s "$json" "$JSON_DIR/"
      done
      cargo docs-md --dir "$JSON_DIR" -o md_docs --exclude-private --source-locations --full-method-docs --hide-trivial-derives
    '';
  };
}
