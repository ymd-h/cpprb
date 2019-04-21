class Animation:
    def add(self,env):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    def display(self,**kwargs):
        raise NotImplementedError()

class NotebookAnimation(Animation):
  def __init__(self):
    self.frames = []

  def add(self,env):
    self.frames.append(env.render(mode='rgb_array'))

  def clear(self):
    self.frames = []

  def display(self,*,dpi = 72,interval=50):
    import matplotlib.pyplot as plt
    from IPython import display
    from matplotlib import animation
    plt.figure(figsize=(self.frames[0].shape[1]/dpi,self.frames[0].shape[0]/dpi),
               dpi=dpi)
    patch = plt.imshow(self.frames[0])
    plt.axis=('off')
    animate = lambda i: patch.set_data(self.frames[i])
    ani = animation.FuncAnimation(plt.gcf(),animate,frames=len(self.frames),
                                  interval=interval)
    return display.HTML(ani.to_jshtml())
