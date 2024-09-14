
def get_mean(values):
    return sum(values) / len(values)

def get_s_xx(x):
    return sum([(x_i - get_mean(x))**2 for x_i in x])

def get_s_yy(y):
    return sum([(y_i - get_mean(y))**2 for y_i in y])

def get_s_xy(x,y):
    return sum([(x_i - get_mean(x))*(y_i - get_mean(y)) for x_i, y_i in zip(x,y)])

def get_sce(x,y):
    return get_s_yy(y) - ((get_s_xy(x,y)**2) / get_s_xx(x))