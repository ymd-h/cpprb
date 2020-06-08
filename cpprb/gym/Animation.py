from abc import ABCMeta, abstractmethod
import os

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
        """

        self._display = None
        if ("DISPLAY" not in os.environ) or (not os.environ["DISPLAY"]):
            from pyvirtualdisplay import Display

            self._display = Display(visible=0, size=size)
            self._display.start()
            _obj = self._display._obj

            # version <=  1.3  : Display._obj.screen
            # version >=  1.3.1: Display._obj._screen
            _screen = _obj.screen if hasattr(_obj,"screen") else _obj._screen
            os.environ["DISPLAY"] = f":{self._display.display}.{_screen}"

        self.frames = []

    def __del__(self):
        """ Destructor

        Stop virtual display if started.
        Delete DISPLAY environment.
        """

        if self._display:
            self._display.stop()
            os.environ["DISPLAY"] = ""

    def add(self,env):
        """
        Add environment into movie frames.

        Parameters
        ----------
        env : gym.Env
            Environment snapshot to be recorded.

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
        """

        if len(self.frames) == 0:
            return

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
        display.display(display.HTML(ani.to_jshtml()))
