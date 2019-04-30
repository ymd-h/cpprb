from abc import ABCMeta, abstractmethod

class Animation(metaclass = ABCMeta):
    @abstractmethod
    def add(self,env):
        raise NotImplementedError()

    @abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abstractmethod
    def display(self,**kwargs):
        raise NotImplementedError()

class NotebookAnimation(Animation):
    """
    Notebook embedded animation class where widget cannot be opened separatedly.

    This class stores frames of display images and shows as HTML movie in notebook.
    """
    def __init__(self,*,size = (1024,768)):
        """Initiate virtual display for Notebook Animation

        Parameters
        ----------
        size : tuple of int, optional
            Display size whose default value is (1024, 768)

        Returns
        -------
        """
        from pyvirtualdisplay import Display
        import os

        display = Display(visible=0, size=size)
        display.start()
        os.environ["DISPLAY"] = ":" + str(display.display) + "." + str(display.screen)

        self.frames = []

    def add(self,env):
        """
        Add environment into movie frames.

        Parameters
        ----------
        env : gym.Env

        Returns
        -------
        """
        self.frames.append(env.render(mode='rgb_array'))

    def clear(self):
        """
        Clear stored environment frames

        Parameters
        ----------

        Returns
        -------
        """
        self.frames = []

    def display(self,*,dpi = 72,interval=50):
        """
        Display environment frames as HTML movie

        Parameters
        ----------
        dpi : int, optional
            dpi for movie whose default value is 72
        interval : int, optional
            interval between frames (frame rate) whose default value is 50

        Returns
        -------
        IPython.display.HTML
            recorded movie

        Notes
        -----
        Returned HTML should be the last line of the cell to display movie or passed
        to display function.
        """
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
