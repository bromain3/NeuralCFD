class FirstOrderInterpolation:

    def interpolate(self, leftInterfaces, rightInterfaces, cells):
        ncells = cells.ncells
        leftInterfaces.Q[:, 1:ncells] = cells.Q[:, 0:ncells-1]
        rightInterfaces.Q[:, 1:ncells] = cells.Q[:, 1:ncells]