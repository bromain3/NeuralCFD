class FirstOrderInterpolation:

    def interpolate(self, leftInterfaces, rightInterfaces, cells):
        leftInterfaces.Q[:, 1:] = cells.Q[:, :] #TODO: compute all interor faces values
        rightInterfaces.Q[:, :-1] = cells.Q[:, :]