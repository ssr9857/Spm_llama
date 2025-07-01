clean:
	cargo clean

build:
	cargo build

test:
	cargo test

lint:
	cargo clippy --all-targets --all-features -- -D warnings

build_release:
	cargo build --release

ios_bindings: build
	cargo run --bin uniffi-bindgen generate --library ./target/debug/libcake.dylib --language swift --out-dir ./cake-ios/bindings
 