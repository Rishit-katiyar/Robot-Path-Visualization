import matplotlib.pyplot as plt
import numpy as np
import heapq
from matplotlib.animation import FuncAnimation

# Define the Cell class to encapsulate properties of each grid cell
class Cell:
    def __init__(self):
        self.parent_i = 0  # Parent cell's row index
        self.parent_j = 0  # Parent cell's column index
        self.f = float('inf')  # Total cost of the cell (g + h)
        self.g = float('inf')  # Cost from start to this cell
        self.h = 0  # Heuristic cost from this cell to destination

# Define colors for visualization
UNBLOCKED_COLOR = '#FFFF99'  # Light Yellow
BLOCKED_COLOR = '#696969'  # Dim Gray
START_COLOR = '#228B22'  # Forest Green
END_COLOR = '#FF6347'  # Tomato

# Define the size of the grid
GRID_ROWS = 50
GRID_COLS = 50

# Global variables for start and end points
start_point = None
end_point = None

# Function to check if a cell is valid (within the grid)
def is_valid(row, col):
    return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS

# Function to check if a cell is unblocked
def is_unblocked(grid, row, col):
    if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]):
        return False
    return grid[row][col] == 1

# Function to check if a cell is the destination
def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

# Function to calculate the heuristic value of a cell (Euclidean distance to destination)
def calculate_h_value(row, col, dest):
    return abs(row - dest[0]) + abs(col - dest[1])

# Function to generate random obstacles in the grid
def generate_obstacles(grid):
    num_obstacles = int(0.2 * GRID_ROWS * GRID_COLS)  # 20% of total cells as obstacles
    for _ in range(num_obstacles):
        row = np.random.randint(0, GRID_ROWS)
        col = np.random.randint(0, GRID_COLS)
        grid[row][col] = 0  # Mark the cell as blocked

# Function to generate a maze-like pattern of obstacles
def generate_maze(grid):
    # Initialize the grid with unblocked cells
    grid.fill(1)

    # Create random walls
    for _ in range(0, int((GRID_ROWS * GRID_COLS) / 10)):
        x = np.random.randint(0, GRID_ROWS)
        y = np.random.randint(0, GRID_COLS)
        grid[x][y] = 0

    # Smooth the walls to create a maze-like pattern
    for _ in range(0, 3):
        for i in range(1, GRID_ROWS - 1):
            for j in range(1, GRID_COLS - 1):
                neighbors = 0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if grid[i + dx][j + dy] == 0:
                            neighbors += 1
                if neighbors > 5:
                    grid[i][j] = 0

# Function to visualize the grid with path
def visualize_path(grid, path, src, dest):
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot the grid with colors
    ax.imshow(grid, cmap='binary', interpolation='nearest')

    # Plot the path
    for i in range(1, len(path)):
        ax.plot([path[i-1][1], path[i][1]], [path[i-1][0], path[i][0]], color='red')

    # Mark the start and end points with labels
    ax.text(src[1], src[0], 'Start', color=START_COLOR, fontsize=12, ha='center')
    ax.text(dest[1], dest[0], 'End', color=END_COLOR, fontsize=12, ha='center')

    # Add title and axis labels
    ax.set_title('A* Pathfinding Visualization')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')

    # Show the plot
    plt.gca().invert_yaxis()  # Invert the y-axis to fix the display orientation
    plt.tight_layout()
    plt.show()

# Function to handle mouse click events for selecting start and end points
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

# A* search algorithm implementation
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

    # Initialize the open list (priority queue)
    open_list = [(0, i, j)]  # (f, row, col)
    heapq.heapify(open_list)

    # Directions of movement: up, down, left, right, diagonal
    row_nbr = [-1, 1, 0, 0, -1, -1, 1, 1]
    col_nbr = [0, 0, -1, 1, -1, 1, -1, 1]

    # Loop until open list is empty
    while open_list:
        # Pop the cell with the minimum f value
        f, i, j = heapq.heappop(open_list)
        closed_list[i][j] = True  # Mark the cell as visited

        # Check if the current cell is the destination
        if is_destination(i, j, dest):
            # Generate the path
            path = []
            while not (i == src[0] and j == src[1]):
                path.append([i, j])
                i, j = cell_details[i][j].parent_i, cell_details[i][j].parent_j
            path.append([i, j])
            path.reverse()
            return path

        # Explore all 8 neighbor cells
        for k in range(8):
            row_next = i + row_nbr[k]
            col_next = j + col_nbr[k]

            # Check if the neighbor is valid, unblocked, and not visited
            if is_valid(row_next, col_next) and is_unblocked(grid, row_next, col_next) and not closed_list[row_next][col_next]:
                # Calculate the g, h, and f values
                g_next = cell_details[i][j].g + 1
                h_next = calculate_h_value(row_next, col_next, dest)
                f_next = g_next + h_next

                # Check if the neighbor is not in the open list or has a lower f value
                if cell_details[row_next][col_next].f == float('inf') or cell_details[row_next][col_next].f > f_next:
                    # Update the details of the neighbor cell
                    cell_details[row_next][col_next].f = f_next
                    cell_details[row_next][col_next].g = g_next
                    cell_details[row_next][col_next].h = h_next
                    cell_details[row_next][col_next].parent_i = i
                    cell_details[row_next][col_next].parent_j = j

                    # Add the neighbor to the open list
                    heapq.heappush(open_list, (f_next, row_next, col_next))

    print("Failed to find the destination cell")
    return []

# Function to animate the pathfinding process
def animate_pathfinding(grid, path, src, dest):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title('A* Pathfinding Visualization')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')

    # Initialize the grid plot
    img = ax.imshow(grid, cmap='binary', interpolation='nearest')

    # Function to update the animation
    def update(frame):
        if frame < len(path):
            x, y = path[frame]
            ax.plot(y, x, marker='o', color='red', markersize=8)
        else:
            ani.event_source.stop()  # Stop the animation when path is fully drawn

    # Animate the pathfinding process
    ani = FuncAnimation(fig, update, frames=len(path) + 5, interval=500)
    plt.gca().invert_yaxis()  # Invert the y-axis to fix the display orientation
    plt.tight_layout()
    plt.show()

# Main function to run the A* search algorithm and visualize the path
def main():
    # Create the grid
    grid = np.ones((GRID_ROWS, GRID_COLS))  # Initialize all cells as unblocked
    generate_maze(grid)  # Generate maze-like obstacles

    # Plot the grid and prompt for selecting start and end points
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title('Select Start and End Points (Left click for start, Right click for end)')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.imshow(grid, cmap='binary', interpolation='nearest')

    # Connect mouse click event
    plt.connect('button_press_event', onclick)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Run the A* search algorithm if both start and end points are selected and within grid boundaries
    if start_point and end_point and is_valid(start_point[0], start_point[1]) and is_valid(end_point[0], end_point[1]):
        path = a_star_search(grid, start_point, end_point)
        if path:
            visualize_path(grid, path, start_point, end_point)
            animate_pathfinding(grid, path, start_point, end_point)
    else:
        print("Both start and end points are required and must be within the grid boundaries.")

# Entry point of the program
if __name__ == "__main__":
    main()
