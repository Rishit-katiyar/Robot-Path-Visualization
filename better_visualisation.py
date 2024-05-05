import matplotlib.pyplot as plt
import numpy as np
import math
import heapq

# Define the Cell class
class Cell:
    def __init__(self):
        self.parent_i = 0  # Parent cell's row index
        self.parent_j = 0  # Parent cell's column index
        self.f = float('inf')  # Total cost of the cell (g + h)
        self.g = float('inf')  # Cost from start to this cell
        self.h = 0  # Heuristic cost from this cell to destination

# Define the size of the grid with some drift
GRID_ROWS = 50 + np.random.randint(-5, 5)  # Introduce drift in row count
GRID_COLS = 50 + np.random.randint(-5, 5)  # Introduce drift in column count

# Define colors for visualization
UNBLOCKED_COLOR = 'white'
BLOCKED_COLOR = 'black'
PATH_COLOR = 'blue'
START_COLOR = 'green'
END_COLOR = 'red'

# Global variables for start and end points
start_point = None
end_point = None

# Check if a cell is valid (within the grid)
def is_valid(row, col):
    return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS

# Check if a cell is unblocked
def is_unblocked(grid, row, col):
    if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]):
        return False
    return grid[row][col] == 1

# Check if a cell is the destination
def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

# Calculate the heuristic value of a cell (Euclidean distance to destination)
def calculate_h_value(row, col, dest):
    return abs(row - dest[0]) + abs(col - dest[1])

# Trace the path from source to destination
def trace_path(cell_details, dest):
    path = []
    row = dest[0]
    col = dest[1]

    # Trace the path from destination to source using parent cells
    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j
        row = temp_row
        col = temp_col

    # Add the source cell to the path
    path.append((row, col))
    # Reverse the path to get the path from source to destination
    path.reverse()

    return path

# Implement the A* search algorithm
def a_star_search(grid, src, dest):
    # Check if the source and destination are valid
    if not is_valid(src[0], src[1]) or not is_valid(dest[0], dest[1]):
        print("Source or destination is invalid")
        return

    # Check if the source and destination are unblocked
    if not is_unblocked(grid, src[0], src[1]) or not is_unblocked(grid, dest[0], dest[1]):
        print("Source or the destination is blocked")
        return

    # Check if we are already at the destination
    if is_destination(src[0], src[1], dest):
        print("We are already at the destination")
        return

    # Initialize the closed list (visited cells)
    closed_list = [[False for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    # Initialize the details of each cell
    cell_details = [[Cell() for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    # Initialize the start cell details
    i = src[0]
    j = src[1]
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j

    # Initialize the open list (cells to be visited) with the start cell
    open_list = []
    heapq.heappush(open_list, (0.0, i, j))

    # Initialize the flag for whether destination is found
    found_dest = False

    # Main loop of A* search algorithm
    while open_list:
        # Pop the cell with the smallest f value from the open list
        p = heapq.heappop(open_list)

        # Mark the cell as visited
        i = p[1]
        j = p[2]
        closed_list[i][j] = True

        # For each direction, check the successors
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]

            # If the successor is valid, unblocked, and not visited
            if is_valid(new_i, new_j) and is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                # If the successor is the destination
                if is_destination(new_i, new_j, dest):
                    # Set the parent of the destination cell
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
                    print("The destination cell is found")
                    # Trace and print the path from source to destination
                    path = trace_path(cell_details, dest)
                    found_dest = True
                    return path
                else:
                    # Calculate the new f, g, and h values
                    g_new = cell_details[i][j].g + 1.0
                    h_new = calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new

                    # If the cell is not in the open list or the new f value is smaller
                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        # Add the cell to the open list
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        # Update the cell details
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j

    # If the destination is not found after visiting all cells
    if not found_dest:
        print("Failed to find the destination cell")
        return []

def visualize_path(grid, path, src, dest):
    # Create a colormap for the path with drift
    cmap = plt.cm.get_cmap('cool')  # Introduce drift by selecting different color map

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Hide the axes
    ax.axis('off')

    # Display the grid without numbers and black lines
    ax.imshow(grid, cmap='binary', interpolation='nearest')

    # Plot the path with varying colors and introduce drift in path visualization
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        # Check if the path has a normal right turn or a curved drift turn
        if abs(start[0] - end[0]) <= 1 and abs(start[1] - end[1]) <= 1:
            ax.plot([start[1], end[1]], [start[0], end[0]], color=cmap(i / len(path)), linewidth=2)
        else:
            ax.plot([start[1], end[1]], [start[0], end[0]], color=cmap(i / len(path)), linewidth=2, linestyle='dashed')

    # Mark the start and end points with circles of varying sizes
    ax.plot(src[1], src[0], 'go', markersize=15, alpha=0.8)  # Start point with green circle
    ax.plot(dest[1], dest[0], 'ro', markersize=20, alpha=0.8)  # End point with red circle

    # Add a title
    ax.set_title('A* Pathfinding Visualization', fontsize=20, fontweight='bold')

    # Show the plot
    plt.show()

def onclick(event):
    global start_point, end_point
    col = int(event.xdata + 0.5)
    row = int(event.ydata + 0.5)

    if event.button == 1:  # Left mouse button for start point
        start_point = [row, col]
        print("Start point selected:", start_point)
    elif event.button == 3:  # Right mouse button for end point
        end_point = [row, col]
        print("End point selected:", end_point)

def main():
    fig, ax = plt.subplots(figsize=(10, 10))
    # Define the grid (1 for unblocked, 0 for blocked) with some drift
    grid = np.random.choice([1, 0], size=(GRID_ROWS, GRID_COLS), p=[0.8, 0.2])  # Introduce drift by randomizing grid

    ax.imshow(grid, cmap='binary', interpolation='nearest')
    ax.set_title('Select Start and End Points (Left click for start, Right click for end)')
    ax.axis('off')
    plt.connect('button_press_event', onclick)
    plt.show()

    # Run the A* search algorithm if both start and end points are selected and within grid boundaries
    if start_point and end_point and is_valid(start_point[0], start_point[1]) and is_valid(end_point[0], end_point[1]):
        path = a_star_search(grid, start_point, end_point)
        if path:
            visualize_path(grid, path, start_point, end_point)
    else:
        print("Both start and end points are required and must be within the grid boundaries.")

if __name__ == "__main__":
    main()
