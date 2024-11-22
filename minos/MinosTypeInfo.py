import numpy as np
import re
from minos.MinosUtil import ternary, str2int 
from copy import deepcopy

class TypeInfoObj():
    """ Auxiliary object used by TypeInfo class. 
        It can be considered as a specific tree
        structure.
    """
    def __init__(self, ):
        self.Name = None
        self.EnumType = "None"
        self.Constants = {"name": [], "value": []}
        self.Length = 1
        self.Type = ""
        self.Bytes = 1
        self.Children = []
        self.Str = ""


class TypeInfo():
    """ Type info for deserializing Minos
        structural data. Use cases:
        - obj = TypeInfo(str)
        - obj = TypeInfo(str, name)
        - obj = TypeInfo(typeInfo, multiplier)

    """
    def __init__(self, *args):

        # initialize basic type data
        self.initType()

        # convert the input header and store it into data variable
        self.data = self.analyze(*args)

    def initType(self,):
        # dictionary for different data types
        self.BuiltIns = {
            "logical": "b4",
            "char": "c1",
            "int8": "i1",
            "uint8": "u1",
            "int16": "i2",
            "uint16": "u2",
            "int32": "i4",
            "uint32": "u4",
            "int64": "i8",
            "uint64": "u8",
            "single": "f4",
            "double": "f8"
        }
        self.abs2type = {self.BuiltIns[k]:k for k in self.BuiltIns}

        # regular expression for different data structures
        self.BasePattern = re.compile(r"(?P<name>[a-zA-Z0-9_]*):?(?P<comment>\^?)(?P<length>\d+)(?P<descr>[buicfd\^]+)(?P<bytes>\d+)")
        self.CommonPattern = re.compile(r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*):?(?P<value>[-\d]+)?")

    def createTokens(self, string):
        # Convert string representation to a token list
        n = len(string)
        
        # Initialize the equivalent of the MATLAB table using a list of dictionaries
        rep = [{'index': 0, 'level': 0, 'c': ''} for _ in range(n)]
        
        level = 1
        max_level = 1

        # Iterate through each character in the string
        for k in range(n):
            c = string[k]
            rep[k]['index'] = k 
            
            if c == "(":
                rep[k]['level'] = level
                rep[k]['c'] = "^"
                level += 1
                if level > max_level:
                    max_level = level
            elif c == ")":
                rep[k]['level'] = level
                rep[k]['c'] = "$"
                level -= 1
            else:
                rep[k]['level'] = level
                rep[k]['c'] = c

        # Create the tokens list
        tokens = [[] for _ in range(max_level)]
        
        for k in range(1, max_level+1):
            # Add rows where level matches current level to tokens[k]
            tokens[k-1] = [row for row in rep if row['level'] == k]
        
        return tokens
    
    def parse(self, tokens, level=1, start=0, length=None):
        """ Parsing the tokenized string.
        """

        if length is None:
            length = len(tokens[level-1]) # note that Python is 0-index instead of 1 for Matlab
        thisLevel = tokens[level-1][start:start+length]
        s = "".join([cur['c'] for cur in thisLevel])
        match = re.match(self.BasePattern, s)

        if match is not None:
            m = match.groupdict()
            extents = []
            for k in m:
                extents.append(match.span(k))
        else:
            assert 0, "Invalid type specification %s"%s        

        
        # initialize the output object
        obj = TypeInfoObj()
        if m['name'] != "":
            obj.Name = str(m['name'][0]).upper()+str(ternary(len(m['name'])>1, m['name'][1:], ''))
        else:
            obj.Name = "" 
        obj.Length = str2int(m['length'])
        obj.Bytes = str2int(m['bytes'])
        names = ""

        # convert comment section
        if m['comment'] == '^':
            index = thisLevel[extents[1][0]]['index']+1
            
            # note that this is equivalent to level+1 in Matlab
            for idx, cur in enumerate(tokens[level]):
                if cur['index'] == index:
                    nextIndex = idx 
                    break

            for idx, cur in enumerate(tokens[level][nextIndex:]):
                if cur['c'] == "$":
                    count = idx # for python, the idx is already offset by 1 thus no need to further -1
                    break

            comment = "".join([cur['c'] for cur in tokens[level][nextIndex:nextIndex+count]])            
            matched_group = re.finditer(self.CommonPattern, comment)
            cs = dict()
            for match in matched_group:
                for k in match.groupdict():
                    if k not in cs:
                        cs[k] = []
                    cs[k].append(match[k])
            assert len(cs)>0, "Invalid comment "+comment

            if cs['value'][0] is None:
                names = cs['name']
            else:
                id = m['descr'] + m['bytes']
                obj.Type = self.abs2type[id]
                obj.Constants['name'] = cs['name']
                obj.Constants['value'] = [str2int(cs['value'][idx], obj.Type) for idx in range(len(cs['value']))]
                obj.EnumType = ternary(obj.Constants['value'][0]!=0, "EnumLike", "Enum")
            
            if comment[0] == '^':
                obj.EnumType = 'Flag'
        
        # convert the description section
        if m['descr'] != '^':
            id = m['descr'] + m['bytes']
            obj.Type = self.abs2type[id]
            if names != "":
                assert obj.Length == len(names), "Comment %s doesn't match field length %d" %(comment, obj.Length)
                ti1 = deepcopy(obj)
                ti1.Length = 1
                for idx in range(obj.Length):
                    ti1.Name = names[idx]
                    ti1 = self.updateString(ti1)
                    obj.Children.append(deepcopy(ti1))
                obj.Length = 1
                obj.Type = ''
                obj.Bytes = obj.Bytes*len(names)
        else:
            children = []
            index = thisLevel[extents[3][0]]['index']+1 
            nextIndex = len([cur['index'] for cur in tokens[level] if cur['index']<index])

            while True:
                for idx, cur in enumerate(tokens[level][nextIndex:]):
                    if cur['c'] in ['$', ',']:
                        count = idx # for python, the idx is already offset by 1 thus no need to further -1
                        break
                children.append(self.updateString(self.parse(tokens, level+1, nextIndex, count)))
                nextIndex += count
                # reach the end of the object (defined by ")")
                if tokens[level][nextIndex]['c'] == '$':
                    break
                nextIndex += 1
            
            if names!="":
                assert len(children)==1, "Invalid comment %s" %comment
                ti1 = deepcopy(children[0])
                ti1.Length = 1
                children = []
                for i in range(len(names)):
                    ti1.Name = names[i]
                    ti1 = self.updateString(ti1)
                    children.append(deepcopy(ti1))
            obj.Children = children
        
        obj = self.updateString(obj)
                    
        return obj
    
    def updateString(self, obj):
        """auxiliary function for updating the string of the 
            output object based on its child.
        """
        # To string
        if obj.EnumType == "None":
            if len(obj.Children) == 0:  # Built-in type or its array
                obj.Str = f"{obj.Length}{self.BuiltIns.get(obj.Type, '')}"
            elif len(obj.Children) > 1:  
                if all(child.Type == obj.Children[0].Type for child in obj.Children):
                    children_names = ",".join(child.Name for child in obj.Children)
                    first_child_str = self.analyze(obj.Children[0], len(obj.Children)).Str
                    obj.Str = f"({children_names}){first_child_str}"
                else:
                    children_str = ",".join(self.selectJoin(
                        lambda x: ("" if x.Name == "" else x.Name + ":") + x.Str, obj.Children))
                    obj.Str = f"1({children_str}){obj.Bytes}"
            else:  # Struct array
                first_child_str = obj.Children[0].Str
                obj.Str = f"{obj.Length}({first_child_str}){obj.Bytes}"
        elif obj.EnumType == "Enum":  # Enum
            const_str = self.constants2str(obj.Constants)
            obj.Str = f"({const_str}){obj.Length}{self.BuiltIns[obj.Type]}"
        elif obj.EnumType == "Flag":  # Flags
            const_str = self.constants2str(obj.Constants)
            self.Str = f"((flags){const_str}){obj.Length}{self.BuiltIns[obj.Type]}"
        elif obj.EnumType == "EnumLike":  # Enum-like
            const_str = self.constants2str(obj.Constants)
            self.Str = f"({const_str}){obj.Length}i4"
        
        return obj
    
    def selectJoin(self, func, in_list, sep=None):
        """
        Applies a function to each element in the input list and joins the results with a separator if provided.
        """
        # Apply the function to each element using a list comprehension
        out = [func(item) for item in in_list]
        
        # Join the results with the separator if provided
        if sep is not None and sep != "":
            out = sep.join(out)
        
        return out
    
    def constants2str(self, constants):
        """Convert Minos Constant object to string.
        """
        string = []
        for i in range(len(constants['name'])):
            string.append('%s:%d' %(constants['name'][i], constants['value'][i]))

        return ','.join(string)

    def analyze(self, *args):
        """ main function for generate the deserialized object 
            based on input length/type. 
        """
        if not args:
            return
        
        # for obj = TypeInfo(str)
        if len(args) == 1 and isinstance(args[0], str):
            tokens = self.createTokens(args[0])
            obj = self.parse(tokens)
        
        # for obj = TypeInfo(str, name)
        # create the typeInfo object and rename it accordingly
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            tokens = self.createTokens(args[0])
            obj = self.parse(tokens)
            obj.Name = args[1]

        # for obj = TypeInfo(typeInfo, multiplier)
        # copying the input typeInfo to a new object
        elif len(args) == 2 and isinstance(args[0], TypeInfoObj) and isinstance(args[1], (int, float)):
                typeInfo = args[0]
                multiplier = args[1]
                obj = TypeInfoObj()
                obj.Length = multiplier
                obj.Bytes = typeInfo.Bytes

                if len(typeInfo.Children)<= 1:
                    obj.EnumType = typeInfo.EnumType
                    obj.Constants = typeInfo.Constants
                    obj.Type = typeInfo.Type
                    obj.Children = typeInfo.Children
                else:
                    obj.Children.append(typeInfo)

                obj = self.updateString(obj)  # Update the string representation        
        return obj

    def flattenChild(self, obj):
        """ Traverse the TypeInfoObj Tree in a depth-first-search manner
            and flatten all nodes into a sequence.
        """
        if len(obj.Children) == 0:
            return [obj]
        else:
            n = len(obj.Children)
            objs = []
            for i in range(n):
                objs.extend(self.flattenChild(obj.Children[i]))
            
            # adjust the name of leaf node if applicable
            isTop = (obj.Name is None)
            for i in range(len(objs)):
                if objs[i].Name is None:
                    name = "_"
                elif not isTop and len(objs[i].Name) == 1:
                    name = obj.Name + objs[i].Name.upper()
                elif not isTop:
                    name = obj.Name + objs[i].Name[0].upper() + objs[i].Name[1:]
                else:
                    name = objs[i].Name
                objs[i].Name = name

            return objs
    
    def getFormat(self, ):
        """ Flatten the TypeInfoObj tree and generate 
            the format information for reading binary
            files.
        """
        objs = self.flattenChild(self.data)

        # reformatting the format infor as a dictionary instead of cell in Matlab
        fmt = {'Type': [], 'Size': [], 'Name': []}
        for i in range(len(objs)):
            t = objs[i].Type
            if t == 'logical':
                t = 'int32'
            elif t == 'char':
                t = 'uint8'
            
            fmt['Type'].append(t)
            fmt['Size'].append([1, objs[i].Length])
            fmt['Name'].append(objs[i].Name)

        return fmt, objs

    def decode(self, obj, values):
        """ Decode the value of the binary files based on Typeinfo constants.
        """
        # assuming values is stored in a numpy array
        n = values.size 
        values = values.reshape(-1)
        if obj.EnumType == 'Flag':
            out = []
            isCode = ((values[:, np.newaxis] & np.array(obj.Constants['value'])) > 0).astype(int)
            for i in range(n):
                # the final value can be a combination of multiple base values (indicated in each row in isCode)
                # concatenate the selected base values with _
                out.append('_'.join([obj.Constants['name'][cur] 
                            for cur in range(len(obj.Constants['name'])) if isCode[i, cur]]))
        elif obj.EnumType == 'Enum':
            isCode = (values[:, np.newaxis] == np.array(obj.Constants['value'])).astype(int) 
            assert np.all(np.sum(isCode, 1)>0), 'Undecodable value'
            # only one value will be valid
            idc = np.argmax(isCode, axis=1) 
            out = [obj.Constants['name'][cur] for cur in idc]
        else:
            out = values

        return out

                