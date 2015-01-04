class Config(object):
    """
    Class that loads options from a config file and converts
    them into attributes. 
    """
    def __init__(self, filename):
        config = {}
        config_file = open(filename)
        config_text = config_file.read()
        config_file.close()
        if "config" not in config_text:
            raise ValueError
        exec config_text
        for a, v in config.iteritems():
            setattr(self, a, v)
 
