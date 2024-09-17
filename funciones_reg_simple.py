
def get_mean(values):
    return sum(values) / len(values)

def get_s_xx(x):
    return sum([(x_i)**2 for x_i in x]) - (sum(x)**2 / len(x))

def get_s_yy(y):
    mean = get_mean(y)
    return sum([(y_i - mean)**2 for y_i in y])

def get_s_xy(x,y):
    return sum([(x_i)*(y_i) for x_i, y_i in zip(x,y)]) - ((sum(x)*sum(y)) / len(x))

def get_sce(x,y):
    return get_s_yy(y) - ((get_s_xy(x,y)**2) / get_s_xx(x))

def get_stc(y):
    return get_s_yy(y)

def get_r_2(x,y):
    return 1 - (get_sce(x,y) / get_stc(y))