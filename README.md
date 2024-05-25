# Mother Machine Tracker

This tool is designed for the precise tracking and analysis of breakpoints within a montage of microscopic images. It allows users to load montages, interactively set and correct breakpoints, and save these data for further analysis. The tool supports loading montages from two different channels (e.g., phase and fluorescent), providing flexibility in visualization and analysis. Users can insert new breakpoints, correct existing ones, or trim unnecessary breakpoints with intuitive mouse interactions. Additionally, the tool allows for the seamless switching between channels, ensuring comprehensive analysis across different imaging modalities.

## Features

- **Interactive Breakpoint Setting:** Users can click to set new breakpoints, drag to adjust, and correct breakpoints with ease.
- **Channel Switching:** Seamlessly switch between different channels (e.g., phase and fluorescent) for comprehensive analysis.
- **Breakpoint Management:** Insert, correct, and trim breakpoints using intuitive mouse interactions.
- **Data Persistence:** Save and load breakpoints to ensure continuity in analysis sessions.

## Usage

To run the tool, execute the following command:
```bash
python3 py_MotherMachine.py
```

## Key Bindings

- **Shift + Click:** Insert new breakpoints.
- **Control + Click:** Trim breakpoints after the clicked frame.
- **Alt + Click:** Correct only the clicked breakpoint.
- **Click:** Correct breakpoints from the clicked frame onwards.
- **Enter:** Save or update breakpoints.
- **+ / -:** Zoom in and out of the montage.
- **Left / Right Arrow:** Move left or right through the montage.
- **Up / Down Arrow:** Navigate through saved breakpoints.
- **Space:** Toggle the display of all breakpoints.
- **c:** Switch between different channels.
- **d:** Delete the current track.
- **Escape:** Save all breakpoints and exit.

## File Structure

- `py_MotherMachine.py`: Main script for running the tool.
- `montagesCFP/`: Directory containing CFP montage files.
- `montagesTIF/`: Directory containing TIF montage files.
- `MontagesPKL/`: Directory where breakpoint data are saved as PKL files.

## Authors

[@Systems Biology Lab, CCG-UNAM](https://github.com/ccg-esb-lab)

## License

[MIT](https://choosealicense.com/licenses/mit/)

This project is licensed under the MIT License - see the [license](LICENSE) file for details. 

## Aknowledgements

We thank the help and input from past and present members of the Systems and Synthetic Laboratory at CCG-UNAM.

