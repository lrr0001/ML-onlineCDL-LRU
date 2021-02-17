import tensorflow as tf

class ADMM(tf.keras.layers.Layer):
    def __init__(self,rho,alpha,noi,*args,**kwargs):
        self.rho = rho
        self.alpha = alpha
        self.noi = noi
        super().__init__(*args,**kwargs)

    # These initializations happen once per input (negC,y,By,u):
    def init_vars(self,s):
        negC = self.get_negative_C(s)
        x,Ax = self.init_x(s,negC)
        y,By = self.init_y(s,x,Ax,negC)
        u = self.init_u(s,Ax,By,negC)
        itstats = self.init_itstats(s)
        return (y,u,By,negC,itstats)
    def init_x(self,s,negC):
        raise NotImplementedError
    def init_y(self,s,x,Ax,negC):
        raise NotImplementedError
    def init_u(self,s,Ax,By,negC):
        raise NotImplementedError
    def get_negative_C(self,s):
        raise NotImplementedError
    def init_itstats(self,s):
        return []


    # iterative steps:
    def xstep(self,y,u,By,negC):
        raise NotImplementedError
    def relax(self,Ax,By,negC):
        raise NotImplementedError
    def ystep(self,x,u,Ax_relaxed,negC):
        raise NotImplementedError
    def ustep(self,u,Ax_relaxed,By,negC):
        raise NotImplementedError
    def itstats_record(self,x,y,u,Ax,By,negC,itstats):
        return itstats
    def solvestep(self,y,u,By,negC,itstats):
        x,Ax = self.xstep(y,u,By,negC)
        Ax_relaxed = self.relax(Ax,By,negC)
        y,By = self.ystep(x,u,Ax_relaxed,negC)
        u = self.ustep(u,Ax_relaxed,By,negC)
        itstats = self.itstats_record(x,y,u,Ax,By,negC,itstats)
        return (y,u,By,itstats)

    # Before and After:
    def preprocess(self,s):
        return s
    def get_output(self,s,y,u,By,negC,itstats):
        x,Ax = self.xstep(y,u,By,negC)
        return (x,itstats)

    # The Call function    
    def call(self,s):
        s = self.preprocess(s)
        y,u,By,negC,itstats = self.init_vars(s)
        for ii in range(self.noi):
            y,u,By,itstats = self.solvestep(y,u,By,negC,itstats)
        return self.get_output(s,y,u,By,negC,itstats)
