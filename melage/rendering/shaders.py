try:
    from OpenGL import NullFunctionError
except ImportError:
    from OpenGL.error import NullFunctionError
from OpenGL.GL import *
from OpenGL.GL import shaders
import re

## For centralizing and managing vertex/fragment shader programs.

def initShaders():
    global Shaders
    Shaders = [
        ShaderProgram(None, []),
        ShaderProgram('pointSprite', [   ## allows specifying point size using normal.x
            ## See:
            ##
            ##  http://stackoverflow.com/questions/9609423/applying-part-of-a-texture-sprite-sheet-texture-map-to-a-point-sprite-in-ios
            ##  http://stackoverflow.com/questions/3497068/textured-points-in-opengl-es-2-0
            ##
            ##
            VertexShader("""
                void main() {
                    gl_FrontColor=gl_Color;
                    gl_PointSize = gl_Normal.x;
                    gl_Position = ftransform();
                } 
            """),
        ]),
    ]


CompiledShaderPrograms = {}
    
def getShaderProgram(name):
    return ShaderProgram.names[name]

class Shader(object):
    def __init__(self, shaderType, code):
        self.shaderType = shaderType
        self.code = code
        self.compiled = None
        
    def shader(self):
        if self.compiled is None:
            try:
                self.compiled = shaders.compileShader(self.code, self.shaderType)
            except NullFunctionError:
                raise Exception("This OpenGL implementation does not support shader programs; many OpenGL features in pyqtgraph will not work.")
            except RuntimeError as exc:
                ## Format compile errors a bit more nicely
                if len(exc.args) == 3:
                    err, code, typ = exc.args
                    if not err.startswith('Shader compile failure'):
                        raise
                    code = code[0].decode('utf_8').split('\n')
                    err, c, msgs = err.partition(':')
                    err = err + '\n'
                    msgs = re.sub('b\'','',msgs)
                    msgs = re.sub('\'$','',msgs)
                    msgs = re.sub('\\\\n','\n',msgs)
                    msgs = msgs.split('\n')
                    errNums = [()] * len(code)
                    for i, msg in enumerate(msgs):
                        msg = msg.strip()
                        if msg == '':
                            continue
                        m = re.match(r'(\d+\:)?\d+\((\d+)\)', msg)
                        if m is not None:
                            line = int(m.groups()[1])
                            errNums[line-1] = errNums[line-1] + (str(i+1),)
                            #code[line-1] = '%d\t%s' % (i+1, code[line-1])
                        err = err + "%d %s\n" % (i+1, msg)
                    errNums = [','.join(n) for n in errNums]
                    maxlen = max(map(len, errNums))
                    code = [errNums[i] + " "*(maxlen-len(errNums[i])) + line for i, line in enumerate(code)]
                    err = err + '\n'.join(code)
                    raise Exception(err)
                else:
                    raise
        return self.compiled

class VertexShader(Shader):
    def __init__(self, code):
        Shader.__init__(self, GL_VERTEX_SHADER, code)
        
class FragmentShader(Shader):
    def __init__(self, code):
        Shader.__init__(self, GL_FRAGMENT_SHADER, code)
        
        
        

class ShaderProgram(object):
    names = {}
    
    def __init__(self, name, shaders, uniforms=None):
        self.name = name
        ShaderProgram.names[name] = self
        self.shaders = shaders
        self.prog = None
        self.blockData = {}
        self.uniformData = {}
        
        ## parse extra options from the shader definition
        if uniforms is not None:
            for k,v in uniforms.items():
                self[k] = v
        
    def setBlockData(self, blockName, data):
        if data is None:
            del self.blockData[blockName]
        else:
            self.blockData[blockName] = data

    def setUniformData(self, uniformName, data):
        if data is None:
            del self.uniformData[uniformName]
        else:
            self.uniformData[uniformName] = data
            
    def __setitem__(self, item, val):
        self.setUniformData(item, val)
        
    def __delitem__(self, item):
        self.setUniformData(item, None)

    def program(self):
        if self.prog is None:
            try:
                compiled = [s.shader() for s in self.shaders]  ## compile all shaders
                self.prog = shaders.compileProgram(*compiled)  ## compile program
            except:
                self.prog = -1
                raise
        return self.prog
        
    def __enter__(self):
        if len(self.shaders) > 0 and self.program() != -1:
            glUseProgram(self.program())
            
            try:
                ## load uniform values into program
                for uniformName, data in self.uniformData.items():
                    loc = self.uniform(uniformName)
                    if loc == -1:
                        raise Exception('Could not find uniform variable "%s"' % uniformName)
                    if type(data)==int:
                        glUniform1i(loc, data)
                    else:
                        glUniform1fv(loc, len(data), data)
                    
                ### bind buffer data to program blocks
                #if len(self.blockData) > 0:
                    #bindPoint = 1
                    #for blockName, data in self.blockData.items():
                        ### Program should have a uniform block declared:
                        ### 
                        ### layout (std140) uniform blockName {
                        ###     vec4 diffuse;
                        ### };
                        
                        ### pick any-old binding point. (there are a limited number of these per-program
                        #bindPoint = 1
                        
                        ### get the block index for a uniform variable in the shader
                        #blockIndex = glGetUniformBlockIndex(self.program(), blockName)
                        
                        ### give the shader block a binding point
                        #glUniformBlockBinding(self.program(), blockIndex, bindPoint)
                        
                        ### create a buffer
                        #buf = glGenBuffers(1)
                        #glBindBuffer(GL_UNIFORM_BUFFER, buf)
                        #glBufferData(GL_UNIFORM_BUFFER, size, data, GL_DYNAMIC_DRAW)
                        ### also possible to use glBufferSubData to fill parts of the buffer
                        
                        ### bind buffer to the same binding point
                        #glBindBufferBase(GL_UNIFORM_BUFFER, bindPoint, buf)
            except:
                glUseProgram(0)
                raise
                    
            
        
    def __exit__(self, *args):
        if len(self.shaders) > 0:
            glUseProgram(0)
        
    def uniform(self, name):
        """Return the location integer for a uniform variable in this program"""
        return glGetUniformLocation(self.program(), name.encode('utf_8'))

    #def uniformBlockInfo(self, blockName):
        #blockIndex = glGetUniformBlockIndex(self.program(), blockName)
        #count = glGetActiveUniformBlockiv(self.program(), blockIndex, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS)
        #indices = []
        #for i in range(count):
            #indices.append(glGetActiveUniformBlockiv(self.program(), blockIndex, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES))
        

initShaders()
