import csv


def load_coordinates_from_csv(csv_filename):
    """
        Reads a CSV file with columns: x-coordinate, Character, y-coordinate.
        Returns a list of tuples (x, y, char).
    """

    coords = []
    with open(csv_filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip the header row
        for row in reader:
            x = int(row[0])
            char = row[1]
            y = int(row[2])
            coords.append((x, y, char))
    return coords



def create_character_grid(coords):
    """
    Given a list of (x, y, char) tuples, build a 2D grid (list of lists)
    where each position (x, y) is filled with the corresponding character.
    Any position not listed in coords is filled with a space.

    Returns the grid as a list of strings, each string representing one row.
    """
    if not coords:
        return []

    # Determine the maximum x and y to size the grid
    max_x = max(coords, key=lambda c: c[0])[0]
    max_y = max(coords, key=lambda c: c[1])[1]

    # Create a 2D array of spaces with dimensions (max_y+1) x (max_x+1)
    # Note: if y=0 is meant to be the top row, you may invert y when placing.
    grid = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # Place each character in the grid
    for (x, y, char) in coords:
        grid[y][x] = char

    # Convert each row (list of characters) into a single string
    # If you want row 0 to be at the top visually, you may need to reverse the grid
    # or invert y during placement, depending on how you interpret "up" vs. "down."
    grid_rows = ["".join(row) for row in grid]
    return grid_rows


def print_grid(grid_rows):
    """
    Prints each row of the grid on a new line.
    """
    for row_str in grid_rows:
        print(row_str)


def main():
    # 1. Load data (in this example, from a CSV file).
    #    If you're pulling data directly from a Google Doc, adapt this part.
    csv_filename = "output.csv"  # Change to your actual file
    coords = load_coordinates_from_csv(csv_filename)

    # 2. Create the grid.
    grid_rows = create_character_grid(coords)

    # 3. Print or otherwise use the grid.
    print_grid(grid_rows)


if __name__ == "__main__":
    main()
