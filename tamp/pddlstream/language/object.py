from collections import namedtuple, defaultdict
from itertools import count
from pddlstream.language.constants import get_parameter_name
#from pddlstream.language.conversion import values_from_objects
from pddlstream.utils import str_from_object, is_hashable

USE_HASH = True
USE_OBJ_STR = True
USE_OPT_STR = True
OPT_PREFIX = '#'
PREFIX_LEN = 1

class Object(object):
    _prefix = 'v'
    _obj_from_id = {}
    _obj_from_value = {}
    _obj_from_name = {}
    def __init__(self, value, stream_instance=None, name=None):
        self.value = value
        self.index = len(Object._obj_from_name)
        if name is None:
            #name = str(value) # TODO: use str for the name when possible
            name = '{}{}'.format(self._prefix, self.index)
        self.pddl = name
        self.stream_instance = stream_instance # TODO: store first created stream instance
        Object._obj_from_id[id(self.value)] = self
        Object._obj_from_name[self.pddl] = self
        if is_hashable(value):
            Object._obj_from_value[self.value] = self
    def is_unique(self):
        return True
    def is_shared(self):
        return False
    @staticmethod
    def from_id(value):
        if id(value) not in Object._obj_from_id:
            return Object(value)
        return Object._obj_from_id[id(value)]
    @staticmethod
    def has_value(value):
        if USE_HASH and not is_hashable(value):
            return id(value) in Object._obj_from_id
        return value in Object._obj_from_value
    @staticmethod
    def from_value(value):
        if USE_HASH and not is_hashable(value):
            return Object.from_id(value)
        if value not in Object._obj_from_value:
            return Object(value)
        return Object._obj_from_value[value]
    @staticmethod
    def from_name(name):
        return Object._obj_from_name[name]
    @staticmethod
    def reset():
        Object._obj_from_id.clear()
        Object._obj_from_value.clear()
        Object._obj_from_name.clear()
    def __lt__(self, other): # For heapq on python3
        return self.index < other.index
    def __repr__(self):
        if USE_OBJ_STR:
            return str_from_object(self.value) # str
        return self.pddl

##################################################

class UniqueOptValue(namedtuple('UniqueOptTuple', ['instance', 'sequence_index', 'output'])):
    @property
    def parameter(self):
        # return self.instance.external.outputs[self.output_index]
        return self.output

class SharedOptValue(namedtuple('SharedOptTuple', ['stream', 'inputs', 'input_objects', 'output'])):
    @property
    def values(self):
        return tuple(obj.value for obj in self.input_objects)
        #return values_from_objects(self.input_objects)

##################################################

class DebugValue(object): # TODO: could just do an object
    _output_counts = defaultdict(count)
    _prefix = '@' # $ | @
    def __init__(self, stream, input_values, output_parameter):
        self.stream = stream
        self.input_values = input_values
        self.output_parameter = output_parameter
        self.index = next(self._output_counts[output_parameter])
    # def __iter__(self):
    #     return self.stream, self.input_values, self.output_parameter
    # def __hash__(self):
    #     return hash(tuple(self)) # self.__class__
    # def __eq__(self, other):
    #     return (self.__class__ == other.__class__) and (tuple(self) == tuple(other))
    def __repr__(self):
        # Can also just return first letter of the prefix
        return '{}{}{}'.format(self._prefix, get_parameter_name(self.output_parameter), self.index)

class SharedDebugValue(namedtuple('SharedDebugValue', ['stream', 'output_parameter'])):
    # TODO: this alone doesn't refining at the shared object level
    _prefix = '&' # $ | @ | &
    def __repr__(self):
        #index = hash(self.stream) % 1000
        #index = self.stream.outputs.index(self.output_parameter) # TODO: self.stream is a str
        #return '{}{}{}'.format(self._prefix, get_parameter_name(self.output_parameter), index)
        #return '{}{}'.format(self._prefix, self.stream)
        return '{}{}'.format(self._prefix, get_parameter_name(self.output_parameter))

##################################################

# TODO: just one object class or have Optimistic extend Object
# TODO: make a parameter class that has access to some underlying value

class OptimisticObject(object):
    _prefix = '{}o'.format(OPT_PREFIX) # $ % #
    _obj_from_inputs = {}
    _obj_from_name = {}
    _count_from_prefix = {}
    def __init__(self, value, param):
        # TODO: store first created instance
        self.value = value
        self.param = param
        self.index = len(OptimisticObject._obj_from_inputs)
        if USE_OPT_STR and isinstance(self.param, UniqueOptValue):
            # TODO: instead just endow UniqueOptValue with a string function
            #parameter = self.param.instance.external.outputs[self.param.output_index]
            parameter = self.param.output
            prefix = get_parameter_name(parameter)[:PREFIX_LEN]
            var_index = next(self._count_from_prefix.setdefault(prefix, count()))
            self.repr_name = '{}{}{}'.format(OPT_PREFIX, prefix, var_index) #self.index)
            self.pddl = self.repr_name
        else:
            self.pddl = '{}{}'.format(self._prefix, self.index)
            self.repr_name = self.pddl
        OptimisticObject._obj_from_inputs[(value, param)] = self
        OptimisticObject._obj_from_name[self.pddl] = self
    def is_unique(self):
        return isinstance(self.param, UniqueOptValue)
    def is_shared(self):
        #return isinstance(self.param, SharedOptValue)
        return not isinstance(self.param, UniqueOptValue) # OptValue
    @staticmethod
    def from_opt(value, param):
        # TODO: make param have a default value?
        key = (value, param)
        if key not in OptimisticObject._obj_from_inputs:
            return OptimisticObject(value, param)
        return OptimisticObject._obj_from_inputs[key]
    @staticmethod
    def from_name(name):
        return OptimisticObject._obj_from_name[name]
    @staticmethod
    def reset():
        OptimisticObject._obj_from_inputs.clear()
        OptimisticObject._obj_from_name.clear()
        OptimisticObject._count_from_prefix.clear()
    def __lt__(self, other): # For heapq on python3
        return self.index < other.index
    def __repr__(self):
        return self.repr_name
        #return repr(self.repr_name) # Prints in quotations
