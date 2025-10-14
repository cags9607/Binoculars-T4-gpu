from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent

setup(
    name="ai-gen-text-binoculars",              # distribution name shown to pip
    version="0.1.6",                            # bump whenever packaging changes
    description="Batch AI-text detection via Falcon* Binoculars (T4/A100 friendly)",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/cags9607/Binoculars-T4-gpu",
    author='DeepSee',
    author_email="ahans1@umd.edu",
    license="BSD-3-Clause",
    license_files=["LICENSE.md"],

    # Be explicit so experimental/* is packaged (plus binoculars/* and optional ai_gen_text/*)
    packages=find_packages(include=["experimental*", "binoculars*", "ai_gen_text*"]),
    include_package_data=True,


    install_requires=[
        "tqdm>=4.66",
        # Add others only if you really want pip to pull them automatically
        "transformers>=4.44",
        "accelerate>=0.33",
        "bitsandbytes>=0.43.1",
        "numpy>=1.26",
    ],
    python_requires=">=3.9",

    # Nice UX: a CLI entry point
    entry_points={
        "console_scripts": [
            "bino-infer=experimental.inference:main",
        ]
    },
)

