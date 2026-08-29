import numpy as np
import pandas as pd

from src.bootstrap import add_amplification


def test_aa_calculation():
    rows=[]
    for scenario, frozen, closed in [("zero",1.0,1.2),("moderate",1.0,1.5)]:
        for branch,value in [("frozen",frozen),("closed",closed)]:
            rows.append({"panel":0,"assignment":1,"response_seed":0,"scenario":scenario,"intervention":"none","round":6,"branch":branch,"D":value})
    out=add_amplification(pd.DataFrame(rows))
    assert np.isclose(out.loc[out.scenario=="moderate","AA"].iloc[0],0.3)
