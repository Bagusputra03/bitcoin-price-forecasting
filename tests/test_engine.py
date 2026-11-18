import numpy as np
from research_engine.elm import ELM
from research_engine.scaling import minmax, inv_scale

def test_scaling_roundtrip():
    x=np.array([[1.],[2.],[3.]])
    xs,mn,mx=minmax(x)
    assert np.allclose(inv_scale(xs,mn,mx),x)

def test_elm_shape():
    X=np.array([[0.,0.],[1.,1.],[.5,.3]])
    y=np.array([0.,1.,.4])
    m=ELM(5,1e2,42).fit(X,y)
    assert m.predict(X).shape==(3,)
