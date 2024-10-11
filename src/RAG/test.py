import os

# Get the directory of the currently running Python file
current_directory = os.path.dirname(os.path.relpath(__file__))

print("Current directory:", current_directory)
