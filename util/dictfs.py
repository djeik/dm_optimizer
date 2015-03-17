import os
import json
import fnmatch

class DictFS(object):
    def __init__(self, path, parent=None):
        if not os.path.exists(path):
            return ValueError('no such directory: %s' % (path,))

        if not os.path.isdir(path):
            return ValueError('not a directory: %s' % (path,))

        self.root = path
        self.parent = parent
        self.refresh()

    def refresh(self):
        self.elements = dict((f, None) for f in os.listdir(self.root))

    def up(self):
        if self.parent is None:
            self.parent = DictFS(os.path.join(self.root, '..'))

        return self.parent

    def contents(self):
        """ Retrieve a list of all files and directories stored in this
            directory.
        """
        return self.elements

    def files(self, with_cache=False):
        """ Retrieve a list of all files (excludes directories) stored in this
            directory.
        """
        return list(self.ifiles(), with_cache=with_cache)

    def ifiles(self, with_cache=False):
        """ Retrieves an iterator of all files (excludes directories) stored
            in this directory.
            If with_cache is True, then any cached DictFS subdirectories will
            be preserved; the iterator will then be of tuples of the form
            (filename, Maybe DictFS).
        """
        if with_cache:
            return ((f, c) for (f, c) in self.elements.items()
                    if os.path.isfile(os.path.join(self.root, f)))
        else:
            return (f for f in self.elements.keys()
                if os.path.isfile(os.path.join(self.root, f)))

    def dfiles(self):
        """ Retrieve a DictFS object listing only the files (excludes
            directories) in this directory.
            Warning: calling `refresh` on the resulting DictFS will
            essentially undo the effect of this function.
        """
        d = DictFS(self.root, self.parent)
        d.elements = dict(self.files(with_cache=True))
        return d

    def dirs(self):
        """ Retrieve a list of all directories (excludes regular files)
            in this directory.
        """
        return list(self.idirs())

    def idirs(self):
        """ Retrieve an iterator over the directories (excludes regular files)
            in this directory.
        """
        return (f for f in self.elements
                if os.path.isdir(os.path.join(self.root, f)))


    def ddirs(self):
        """ Retrieve a DictFS object listing only the directories (excludes
            regular files) in this directory.
            Warning: calling `refresh` on the resulting DictFS will
            essentially undo the effect of this function.
        """
        d = DictFS(self.root, self.parent)
        d.elements = self.dirs()
        return d

    def glob(self, pat):
        """ Retrieve a list of all elements in this directory matching
            the given filename pattern.
        """
        return fnmatch.filter(self.elements, pat)

    def dglob(self, pat):
        """ Retrieve a DictFS object listing only the elements in this
            directory that match the given filename pattern.
        """
        d = DictFS(self.root, self.parent)
        d.elements = self.glob(pat)
        return d

    def _open(self, path, mode):
        return open(os.path.join(self.root, path), mode)

    def open(self, path, mode='r'):
        if path not in self.elements:
            raise ValueError('no such file %s in %s.' % (path, self.root))
        return self._open(path, mode)

    def map(self, f, reducer=None, iterable=False, enter=False):
        """ Apply a function to each element of this directory, and optionally
            apply a function to the resulting list.

            Arguments:
                f (callable):
                    the function to apply to each element.
                reducer (callable, default: None):
                    a function to apply to the list resulting from the map.
                iterable (boolean, default: False):
                    whether to pass an iterable or a list to the reducer;
                    has no effect if reducer is None.
                enter (boolean, default: False):
                    whether to call this DictFS's enter method on the element
                    prior to passing it to `f`.
        """
        if enter:
            action = lambda e: f(enter(e))
        else:
            action = f

        if reducer is None:
            return [action(e) for e in self.elements]
        else:
            if iterable:
                return reducer(action(e) for e in self.elements)
            else:
                return reducer([action(e) for e in self.elements])

    def imap(self, f, enter=False):
        """ A lazy version of map. """
        return self.map(f, reducer=lambda x: x, iterable=True, enter=enter)

    def dfilter(self, p, enter=False):
        """ Return a DictFS whose elements are only those of the current
            DictFS satisfying a given predicate.
            If the `enter` switch is set, then the result of this DictFS's
            `enter` method, instead of the filename, will passed to the
            predicate.
        """
        d = DictFS(self.root, self.parent)
        d.elements = filter(p, self.elements)
        return d

    def enter(self, path):
        """ Get a file in this directory. If the file is a directory, then a
            new DictFS instance is returned for that directory, and its parent
            is set to the current DictFS, effectively linking them.
            Child DictFS objects are cached by their parents, so references
            to existing DictFS objects are returned whenever possible.
        """
        if path not in self.elements:
            raise ValueError('no such file %s in %s.' % (path, self.root))

        if os.path.isfile(os.path.join(self.root, path)):
            return self._open(path, 'r')
        else:
            d = self.elements[path]
            if d is None:
                d = DictFS(os.path.join(self.root, path), parent=self)
                self.elements[path] = d

            return d

    def __getitem__(self, path):
        return self.enter(path)
