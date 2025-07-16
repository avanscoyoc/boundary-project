# boundary-project

## Overview
This project contains an analysis of habitat continuity and islandization in protected areas using MODIS and geospatial methods. The `example.ipynb` in the notebooks folder includes code for a step-by-step walk through of the analysis; this includes protected area filtering, MODIS imagery retrieval, geometry operations, data processing, gradient calculation, result export, and visualization functionality. The `src` folder contains the application to run the full analysis and includes functions called in the `analysis.py` analysis and `example.ipynb` scripts. The working manuscript is located in the `docs` folder.

## Project Structure
```
boundary-project
├── .devcontainer             # startup container 
├── docs                      # manuscript, references
├── images                    # supporting graphics
├── notebooks
│   ├── example.ipynb         # example workflow, step-by-step 
│   └── visualization.ipynb   # functions for visualizations
├── output                    # outputs for example workflow
├── src
│   ├── analysis.py           # 3. full workflow function, iteration function 
│   ├── config.py             # 1. functions to load/filter protected area data
│   ├── main.py               # 4. main execution function
│   └── utils.py              # 2. all geometric, image, feature, export, viz functions
├── myst.yml                  # latex frontmatter
└── README.md
```

## Setup Instructions

### Prerequisites
- VS code installed on your machine (if running locally)
- Docker installed on your machine (if running locally)
- Access to a JupyterHub instance (if running on a browser)

### Using with VS code
Clone the repository, open VS code, when prompted select to `Open in Container` or in the command palette select `Development Environment: Reopen in Container`. Wait for the container to build, authentic to Earth Engine following the steps below, then select the jupyter kernel and run the example.ipynb notebook.

### Dependencies
Dependencies are automatically installed when building the Docker image.

## Google Earth Engine Authentication
When opening the container for the first time, you'll need to authenticate with Google Earth Engine:
1. VS code's terminal will prompt you to authenticate when you open the container
2. Follow the authentication link provided
3. Copy and paste the authentication token back into the terminal

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.