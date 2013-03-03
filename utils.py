def ipython_on_error():
    import sys
    import IPython
    def excepthook(type, value, traceback):
        print value
        IPython.embed()
    sys.excepthook = excepthook
