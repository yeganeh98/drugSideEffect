class AverageMeter:
    def __init__(self):
        self.val = 0
        self.num = 0

    def reset(self):
        self.val = 0
        self.num = 0

    def update(self, val, num):
        self.val += val
        self.num += num

    def avg(self):
        return self.val / self.num
