.PHONY: doctor build test bench lint serve clean

doctor:
	bash scripts/doctor.sh
	python -m gpubook.validate

build:
	cmake --preset release
	cmake --build --preset release -j

test:
	ctest --preset release --output-on-failure
	pytest -q

bench:
	bash scripts/run_benchmarks.sh

lint:
	pre-commit run --all-files

clean:
	rm -rf build results/tmp