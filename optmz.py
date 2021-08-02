import tensorflow as tf

class ADMM(tf.keras.layers.Layer):
    def __init__(self,rho,alpha,noi,*args,**kwargs):
        self.rho = rho
#        self.alpha = alpha
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
    def itstats_record(self,x,y,u,Ax,Ax_relaxed,By,negC,itstats):
        return itstats
    def solvestep(self,y,u,By,negC,itstats):
        x,Ax = self.xstep(y,u,By,negC)
        Ax_relaxed = self.relax(Ax,By,negC)
        y,By = self.ystep(x,u,Ax_relaxed,negC)
        u = self.ustep(u,Ax_relaxed,By,negC)
        itstats = self.itstats_record(x,y,u,Ax,Ax_relaxed,By,negC,itstats)
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

class ADMM_Relaxed(tf.keras.layers.Layer):
    def __init__(self,rho,alpha,noi,*args,**kwargs):
        self.rho = rho
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
    def relax(self,u,Ax,By,negC):
        raise NotImplementedError
    def ystep(self,x,u,Ax,negC):
        raise NotImplementedError
    def ustep(self,u,Ax,By,negC):
        raise NotImplementedError
    def itstats_record(self,x,y,u,Ax,By,negC,itstats):
        return itstats
    def solvestep(self,y,u,By,negC,itstats):
        x,Ax = self.xstep(y,u,By,negC)
        u = self.relax(u,Ax,By,negC)
        y,By = self.ystep(x,u,Ax,negC)
        u = self.ustep(u,Ax,By,negC)
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


class FISTA(tf.keras.layers.Layer):
    def __init__(self,L,noi,momweight,*args,**kwargs):
        self.L = L
        self.noi = noi
        self.momweight = momweight
        super().__init__(*args,**kwargs)

    # These initializations happen once per input (negC,y,By,u):
    def init_vars(self,r,s):
        negC = self.get_negative_C(s)
        z = self.init_z(s)
        x = self.init_x(s,z)
        r = self.init_r()
        itstats = self.init_itstats(r,s,x,z)
        return r,x,z,itstats

    def get_negative_C(self,s):
        raise NotImplementedError
    def init_z(self,s):
        raise NotImplementedError
    def init_x(self,s,z):
        raise NotImplementedError
    def init_r(self):
        r = 1.
    def init_itstats(self,s,x,z):
        return []

    # iterative steps:
    def gradstep(self,x):
        raise NotImplementedError
    def proxstep(self,z):
        raise NotImplementedError
    def momstep(self,r,x,z):
        raise NotImplementedError
    def itstats_record(self,r,x,z,itstats):
        return itstats
    def solvestep(self,r,s,x,z,itstats):
        z = self.gradstep(x)
        z = self.proxstep(z)
        r,x = self.momstep(r,x,z)
        itstats = self.itstats_record(r,x,z,itstats)
        return r,x,z,itstats

    # Before and After:
    def preprocess(self,s):
        return s
    def get_output(self,r,s,x,z,itstats):
        raise NotImplementedError

    # The Call function    
    def call(self,r,s):
        s = self.preprocess(s)
        r,x,z,itstats = self.init_vars(r,s)
        for ii in range(self.noi):
            r,x,z,itstats = self.solvestep(r,s,x,z,itstats)
        return self.get_output(r,s,x,z,itstats)

