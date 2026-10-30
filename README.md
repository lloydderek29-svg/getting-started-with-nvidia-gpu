# Getting Started with NVIDIA GPUs

### Advanced Techniques for AI Acceleration with Python, C++, CUDA, and GPU Optimization

This is the companion repository for the book *Moving On with NVIDIA GPUs*, the second edition of *Getting Started with NVIDIA GPUs: A Beginner's Guide to AI Acceleration with Python, C++ and CUDA* by Derek Lloyd. Every program, configuration file, and lab in the book has a home here, organized the way a real engineering project would be.

## About This Repository

The first edition taught you how to get a GPU working. This edition teaches you how to make it work well, how to make it work the same way every time, and how to ship something other people can depend on. The code here follows that idea. It is structured like a professional project, with shared code kept in one place, tests beside the code they check, and results saved with the evidence that explains them.

You do not need to read the whole repository to get value from it. Each chapter has its own folder, and each folder has its own README that explains what is inside, how to run it, and what to expect. This page is only the map.

## If You Have Not Read the First Edition

We encourage you to start with the first edition, [Getting Started with NVIDIA GPUs](https://www.amazon.com/Getting-Started-NVIDIA-GPUs-Acceleration-ebook/dp/B0DD3TSGNP/ref=sr_1_4?crid=33WNKSXDSB2TJ&dib=eyJ2IjoiMSJ9.Cu6W7SaigPFP8DjQZYMIolCUHveJrjweFXRMAMzXebvBXY_pjL4C_1Wm0BAUCPJYrpA2JtrmRHznw3OXupo-b9S3vR_Do-UUMr2zjEPt2z3kqz_njuJDRh7OdZl8YcQastw_T0pWamU54NXDDZhK4lJ7NKXQyXFavk66xLulhfWCdTufYLedbG7HHgaiIRnmb3R1eEQYs8j3pjjxrfrxIRZLbJfXU4IhCvhIvRGXsts.UXLhIo6hi9uvM6BajYLNY-rC_KYcLVbmoylYDEQVkSc&dib_tag=se&keywords=nvidia+gpus&qid=1792346270&s=books&sprefix=nvi%2Cstripbooks%2C1743&sr=1-4). It covers the hardware vocabulary, the driver and toolkit installation, your first CUDA kernel, and the introductory training and inference labs that this book builds on. Where an idea from that book is worth a reminder, the chapters point you back to the relevant place. Keep it close while you work through this one.

## Who This Is For

This repository suits you if you can already run a model on a GPU and now want to understand what limits it, speed it up, and deploy it with confidence. You should be comfortable with a command line, with Python, and with reading basic C++. You do not need to be an expert in any of them, and the book explains each technique as it appears.

## How the Book Is Organized

The book has three parts, and the repository follows the same order.

### Part 1: GPU Infrastructure and Advanced CUDA Programming

Understand your hardware, build an environment you can rebuild at any time, write correct and fast CUDA code, and learn to measure before you change anything.

| Chapter | Topic | Code and instructions |
|---|---|---|
| 1 | NVIDIA GPU Architecture and Performance Planning | [chapters/ch01_architecture](chapters/ch01_architecture/README.md) |
| 2 | Production-Ready CUDA and Python Environments | [chapters/ch02_environments](chapters/ch02_environments/README.md) |
| 3 | Advanced CUDA C++ Programming and Memory Management | [chapters/ch03_advanced_cuda](chapters/ch03_advanced_cuda/README.md) |
| 4 | GPU Profiling and Performance Optimization | [chapters/ch04_profiling](chapters/ch04_profiling/README.md) |

### Part 2: Accelerated Machine Learning and Generative AI

Apply the foundations to training pipelines, computer vision, language models, and image generation.

| Chapter | Topic | Code and instructions |
|---|---|---|
| 5 | High-Performance Deep Learning with PyTorch | [chapters/ch05_training](chapters/ch05_training/README.md) |
| 6 | Computer Vision and GPU-Accelerated Object Detection | [chapters/ch06_vision](chapters/ch06_vision/README.md) |
| 7 | LLM Inference and Memory Optimization | [chapters/ch07_llm](chapters/ch07_llm/README.md) |
| 8 | Generative AI and Efficient Model Fine-Tuning | [chapters/ch08_generative](chapters/ch08_generative/README.md) |

### Part 3: Optimized Inference, Deployment, and Edge AI

Turn trained models into fast, tested, observable services.

| Chapter | Topic | Code and instructions |
|---|---|---|
| 9 | TensorRT, ONNX, and Optimized Model Inference | [chapters/ch09_tensorrt](chapters/ch09_tensorrt/README.md) |
| 10 | Deploying GPU Models with Triton Inference Server | [chapters/ch10_triton](chapters/ch10_triton/README.md) |

The book's Summary, Conclusion, and Appendices A through H are reference material and have no code of their own. The templates and checklists they describe are reflected in the shared folders listed below.

## A Map of the Repository

| Folder | What it holds |
|---|---|
| `chapters` | One folder per chapter, each with its own README, source code, and tests |
| `common` | Code shared by several chapters, so nothing is copied from place to place |
| `configs` | Settings that describe experiments and benchmarks as data |
| `docker` | The files that build the development container |
| `requirements` | The lists of packages each part of the book needs |
| `scripts` | Helper scripts for checking the environment, building, and benchmarking |

## Suggested Ways to Use It

**Read and run in order.** The chapters build on each other, and later labs reuse files that earlier ones produce. For example, the classifier you train in Chapter 6 is the model you optimize in Chapter 9 and serve in Chapter 10. This is the best path if you are working through the book from start to finish.

**Jump to a topic.** Each chapter README lists what it needs before you begin, so you can start at the chapter that matches your problem. If a chapter depends on files from an earlier one, its README says so at the top.

**Use it as a reference.** The shared code and the templates in the appendices are meant to be copied into your own projects. Take what is useful.

## Before You Begin

You will need a computer with a recent NVIDIA graphics card, running Linux or Windows with the Linux subsystem. The book's introduction and Chapter 2 explain the full requirements and the setup in detail, and each chapter README repeats what that chapter needs. If something does not work, the troubleshooting section at the end of every chapter README is the first place to look, and Appendix D of the book collects the most common problems in one place.

## Quick Start

Clone the repository and install the base packages. Run every chapter command from the repository root.

```bash
git clone https://github.com/lloydderek29-svg/getting-started-with-nvidia-gpu.git
cd moving-on-with-nvidia-gpu
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements/base.txt
pip install -r requirements/dev.txt
pip install -e .
make doctor
```

The last command checks your driver, compiler, and Python setup and reports anything that is missing. Each chapter README lists the extra packages that chapter needs, such as the PyTorch stack for Chapters 5 to 10.

## A Note on Results

Your numbers will not match the ones printed in the book. Speed depends on your card, your drivers, your cooling, and the versions of the software you use, and that is expected. What should carry over is the shape of the results: which approach is faster, which uses less memory, and roughly by how much. The code saves the details of your environment next to every measurement, so you can always tell which machine produced which number.

## Keeping Up With Change

The software around NVIDIA GPUs changes every few months. Library versions, model names, and download locations move, and some of the details in this repository will need small updates over time. When something has changed, the book and the chapter READMEs tell you where to look for the current information. The official documentation listed in Appendix H of the book is always the final word.

## Getting Help

If you find a mistake, have a question about a lab, or want to suggest an improvement, open an issue in this repository. Please include which chapter you were working on, what you expected to happen, and what happened instead. The more detail you share about your setup, the easier it is to help.

## Contributing

Contributions are welcome, especially corrections, clearer explanations, and fixes for changes in the libraries the book depends on. Please keep changes focused, test them in the relevant chapter, and describe what you changed and why.

## License

The license for this repository is in the LICENSE file at the top level. Please read it before reusing the code.

## Trademarks and Disclaimer

NVIDIA, the NVIDIA logo, CUDA, GeForce, Jetson, TensorRT, and Triton are trademarks or registered trademarks of NVIDIA Corporation in the United States and other countries. All other trademarks are the property of their respective owners. This repository is not affiliated with or endorsed by NVIDIA.

The code is provided as is, without warranty of any kind. The field of artificial intelligence and GPU hardware moves quickly, and specific version numbers and software behavior may change after publication. Test everything in your own environment before relying on it.

## Acknowledgments

This work stands on the shoulders of the open-source communities and the engineers who build the tools used throughout the book, including the teams behind CUDA, PyTorch, Hugging Face, ONNX, and many others. Thank you to everyone who documents, debugs, and shares what they learn.

Happy building.