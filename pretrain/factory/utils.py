class AverageStdMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.sum_sq += (val * val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

    @property
    def std(self):
        mean = self.avg
        var = self.sum_sq / self.count - mean * mean
        return max(var, 0.0) ** 0.5