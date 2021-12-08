import pandas as pd
import numpy as np

if __name__ == "__main__":
    # 时序数据
    time = pd.Series(np.random.randn(8), index=pd.date_range('2018-06-01', periods=8))
    print(time)
