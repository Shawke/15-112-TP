# This code is Sam Hawke's (shawke) 15-112 term project.
# It is a mathematical software that uses Tkinter to graph functions,
# does high-computation matrix algebra, and does a little bit of statistics.
# I use the barebones timer, mouse, and keyboard events from the 112 course website:
# # citing barebones from 15-112 website:
# https://www.cs.cmu.edu/~112/notes/events-example0.py

# events-example0.py
# Barebones timer, mouse, and keyboard events

# Christine Zhang (my mentor) was highly influential in choosing these colors.
# Big thanks to her for input on design!!!

# RGB COLOR VALUES FROM MY PALETTE:
# (R,G,B) = (241,241,241) white
# (51,51,51) black
# (17,82,164) blue
# (255,210,99) yellow
# (252,104,100) peach

from __future__ import print_function
from __future__ import division
import copy,math

# These are the math functions that the matrices options require

def make2dList(rows, cols, entries=0):
    newList = []
    for row in range(rows):
        newList += [[entries] * cols]
    return newList

def matrixTranspose(A): # changes rows and cols
    rows = len(A)
    cols = len(A[0])
    transposeMatrix = make2dList(cols,rows)
    for col in range(cols):
        for row in range(rows):
            transposeMatrix[col][row] = A[row][col]
    return transposeMatrix

def matrixMultiplier(A,B): # represent A, B as two 2-d lists
    # Consider A an mxn matrix, and B an nxp matrix.
    rowsA = len(A)
    rowsB = len(B)
    colsA = len(A[0])
    colsB = len(B[0])
    if colsA != rowsB:
        return "You can't multiply these matrices together!"
    m = rowsA
    n = rowsB
    p = colsB
    resultingMatrix = make2dList(m, p, "")
    for row in range(m):
        for col in range(p): # these are rows and cols of the resulting matrix
            newEntry = 0
            for colA in range(colsA):
                rowB = colA
                newEntry += A[row][colA] * B[rowB][col]
            resultingMatrix[row][col] = newEntry
    return resultingMatrix

# Big thanks to Miranda Chen (yinglanc) for helping me scaffold
# the Gaussian Elimination function!!!

def gaussianElimination(A): # woohoo!
    # this function should work NON-DESTRUCTIVELY
    B = copy.deepcopy(A)
    rows = len(B)
    cols = len(B[0])
    for col in range(cols):
        # find the first nonzero entry in this column
        # (starting with the row with the same index as the col)
        FNZE = -1
        for row in range(col,rows):
            if B[row][col] != 0:
                FNZE = row
                break
        # if all zeros: continue and move on to the next column
        if FNZE == -1: continue
        # swap the the row of that first nonzero entry with the current row,
        # where the current row is the same index as col
        for c in range(cols):
            B[FNZE][c],B[col][c] = B[col][c],B[FNZE][c]
        # make all rows below this one 0 in this column (col)
        for row in range(col+1,rows):
            factor = B[row][col] / B[col][col]
            for c in range(col,cols):
                B[row][c] = B[row][c] - B[col][c] * factor
    for row in range(len(B)):
        for col in range(len(B[0])):
            B[row][col] = round(B[row][col],3)
    return B

def systemsForSquareMatrices(A,b): # here, we try to solve Ax = b
    # for a given (square) matrix A, and a given target vector b.
    # Here, A will be represented by an nxn 2-d list (as usual),
    # and b will be represented by a 1-d list of length n.
    # Also note: this function will take full advantage of the mathematical
    # insight that is Cramer's explicit formula, so as to avoid 
    # Gaussian elimination (for now).
    n = len(A)
    if n != len(A[0]): return "Don't use this function for nonsquare matrices!"
    if det(A) == 0: return "We have many solutions!" # but can't find them yet
    x = [0] * n # our solution vector
    for i in range(n): # modify our x with Cramer's formula
        # alter the matrix
        newA = copy.deepcopy(A)
        for row in range(n):
            newA[row][i] = b[row]
        xi = det(newA) / det(A)
        x[i] = xi
    return x

def inverse(A): # here we find the inverse of A (if possible)
    m = len(A)
    n = len(A[0])
    if m != n: return "Would you like a 'pseudo'-inverse?" # lol
    if det(A) == 0: return "A is not invertible"
    inverseMatrix = make2dList(n,n)
    for col in range(n):
        unitVector = [0] * n
        unitVector[col] = 1
        solutionVector = systemsForSquareMatrices(A,unitVector)
        for row in range(n):
            inverseMatrix[row][col] = solutionVector[row]
    return inverseMatrix

def det(A): # this will involve recursion!
    n = len(A)
    m = len(A[0])
    if n != m: return "Only square matrices have determinants!"
    if n == 1: return A[0][0]
    if n == 2: return (A[0][0] * A[1][1]) - (A[1][0] * A[0][1])
    else:
        # expand upon the top row
        result = 0
        newA = A[1:] # get the submatrix (minor), whose det we want
        for col in range(n):
            newerA = copy.deepcopy(newA) # to avoid list aliasing
            for row in range(n-1):
                newerA[row] = newA[row][:col] + newA[row][col+1:]
            result += (-1)**(col%2) * A[0][col] * det(newerA)
        return result

from Tkinter import *

####################################
# init
####################################

def init(data): # 90 lines: Holy moly!
    data.helpxValuesV = []
    data.helpxValuesH = []
    data.helpyValuesV = []
    data.helpyValuesH = []
    sep = 8
    for i in range(-40,41):
        data.helpxValuesV += [[data.width//2+i*sep,data.width//2+i*sep]]
        data.helpyValuesV += [[0, data.height]]
        data.helpxValuesH += [[0, data.width]]
        data.helpyValuesH += [[data.height//2+i*sep,data.height//2+i*sep]]
    data.mode = "menu"
    data.count = 0
    data.matrixMode = None
    data.matricesOptions = False
    data.graphingOptions = False
    data.statsOptions = False
    data.help = False
    data.margin = 10
    data.boxHeight = 5*data.height//6-3*data.margin
    data.startHeight = data.height//6+2*data.margin
    data.startHeight2 = data.boxHeight//2+data.startHeight+data.margin
    data.matrixAlist = []
    data.matrixBlist = []
    data.result = []
    data.neg = False
    data.matrix = "A"
    data.Arows = 0
    data.Acols = 0
    data.Bcols = 0
    data.tracker = 0
    data.matricesOptionsList = ["matrix multiplication","gaussian elimination",
                                "determinants calculator","inverse",
                                "solve Ax = b","LU decomposition",
                                "Gram-Schmidt","QR Decomposition"]
    data.colorList = ['#%02x%02x%02x' % (241,241,241), '#%02x%02x%02x' % (51,51,51),
    '#%02x%02x%02x' % (17,82,164),'#%02x%02x%02x' % (255,210,99),'#%02x%02x%02x' % (252,104,100)]
    data.white = data.colorList[0]
    data.black = data.colorList[1]
    data.blue = data.colorList[2]
    data.yellow = data.colorList[3]
    data.peach = data.colorList[4]
    data.dim = 0
    data.det = 0
    data.msg = ""
    data.targetVector = []
    data.solution = []
    data.statsMode = "menu"
    data.univariateData = []
    data.entry = 0
    data.mean = 0
    data.SD = 0
    data.median = 0
    data.min = 0
    data.minFX = 0
    data.max = 0
    data.maxFX = 0
    data.point = False
    data.Q1,data.Q3 = 0,0
    data.L1,data.L2 = [],[]
    data.L1squared,data.L2squared,data.L1L2 = [],[],[]
    data.outliers = []
    data.r = 0
    data.b0 = 0
    data.b1 = 0
    data.graphMode = None
    data.xmax,data.ymax,data.xstep,data.ystep = 13,13,3,3
    data.fct = ["","",""]
    data.graph = False
    data.dum = 0
    data.vectorList = [[1,0],[0,1]] # just for testing
    data.finalVectorList = copy.deepcopy(data.vectorList)
    data.newXvaluesV = []
    data.newYvaluesV = []
    data.newXvaluesH = []
    data.newYvaluesH = []
    data.xValuesV = [] # V for vertical lines
    data.yValuesV = []
    data.yValuesH = [] # H for horizontal lines
    data.xValuesH = []
    sep = 45
    for i in range(-20,21):
        data.xValuesV += [[data.width//2+i*sep,data.width//2+i*sep]]
        data.yValuesV += [[0, data.height]]
        data.xValuesH += [[0, data.width]]
        data.yValuesH += [[data.height//2+i*sep,data.height//2+i*sep]]
    data.finalXvaluesV = []
    data.finalYvaluesV = []
    data.finalXvaluesH = []
    data.finalYvaluesH = []
    data.start = []

######
# mode dispatcher
######

# it can't hurt to say that the idea of a mode dispatcher came from 
# the 112 course website: 
# https://www.cs.cmu.edu/~112/notes/notes-animations-examples.html#modeDemo

def mousePressed(event, data):
    if data.mode == "menu": menuMousePressed(event,data)
    elif data.mode == "matricesOptions": matricesMousePressed(event, data)
    elif data.mode == "statsOptions": statsMousePressed(event, data)
    elif data.mode == "graphingOptions": graphMousePressed(event,data)
    elif data.mode == "help": helpMousePressed(event,data)
    elif data.mode == "matrixMult": matrixMultMousePressed(event, data)

def keyPressed(event, data):
    if data.mode == "menu": menuKeyPressed(event,data)
    elif data.mode == "matricesOptions": matricesKeyPressed(event, data)
    elif data.mode == "statsOptions": statsKeyPressed(event, data)
    elif data.mode == "graphingOptions": graphKeyPressed(event,data)
    elif data.mode == "help": helpKeyPressed(event,data)
    elif data.mode == "matrixMult": matrixMultKeyPressed(event,data)

def timerFired(data):
    if data.mode == "menu": menuTimer(data)
    elif data.mode == "matricesOptions": matricesTimer(data)
    elif data.mode == "statsOptions": statsTimer(data)
    elif data.mode == "graphingOptions": graphTimer(data)
    elif data.mode == "help": helpTimer(data)
    
def redrawAll(canvas, data):
    if data.mode == "menu": menuRedrawAll(canvas, data)
    elif data.mode == "matricesOptions": matricesRedrawAll(canvas, data)
    elif data.mode == "statsOptions": statsRedrawAll(canvas, data)
    elif data.mode == "graphingOptions": graphRedrawAll(canvas,data)
    elif data.mode == "help": helpRedrawAll(canvas,data)
    
############
# menu functions!
############

def menuMousePressed(event, data):
    if inMatricesBox(event.x,event.y,data):
        data.mode = "matricesOptions"
        data.matrixMode = "home"
    if inGraphingBox(event.x,event.y,data):
        data.mode = "graphingOptions"
    if inHelpBox(event.x,event.y,data):
        data.mode = "help"
    if inStatsBox(event.x,event.y,data):
        data.mode = "statsOptions"

def inMatricesBox(x,y,data):
    if x > data.margin and x < data.width//2 - data.margin:
        if y > data.startHeight and y < data.startHeight2 - 2*data.margin:
            return True
    return False

def inGraphingBox(x,y,data):
    if x > data.width//2 + data.margin and x < data.width - data.margin:
        if y > data.startHeight and y < data.startHeight2 - 2*data.margin:
            return True
    return False

def inStatsBox(x,y,data):
    if x > data.margin and x < data.width//2 - data.margin:
        if y > data.startHeight2 and y < data.height - data.margin:
            return True
    return False

def inHelpBox(x,y,data):
    if x > data.width//2 + data.margin and x < data.width - data.margin:
        if y > data.startHeight2 and y < data.height - data.margin:
            return True
    return False

def menuTimer(data):
    pass

def menuKeyPressed(event,data):
    pass

def menuRedrawAll(canvas, data):
    canvas.create_rectangle(0,0,data.width,data.height,fill=data.white,width=0)
    canvas.create_text(data.width//2,data.height//8,
                        text="M a t r X   C a r t e s i a",font="Arial 24 bold",
                        fill=data.black)
    margin = data.margin
    boxHeight = data.boxHeight
    startHeight = data.startHeight
    canvas.create_rectangle(margin,startHeight,data.width//2-margin,
                            boxHeight//2+startHeight-margin,fill=data.blue,width=0)
    canvas.create_text(data.width//4,startHeight+boxHeight//4-margin//2,
                        text="matrices options",font="Arial 13",fill=data.white)
    canvas.create_rectangle(data.width//2+margin,startHeight,
                            data.width-margin,
                            boxHeight//2+startHeight-margin,fill=data.peach,width=0)
    canvas.create_text(3*data.width//4,startHeight+boxHeight//4-margin//2,
                        text="graphing",font="Arial 13",fill=data.white)
    startHeight2 = data.startHeight2
    canvas.create_rectangle(margin,startHeight2,data.width//2-margin,
                            startHeight2+boxHeight//2-margin,fill=data.yellow,width=0)
    canvas.create_text(data.width//4,startHeight2+boxHeight//4-margin//2,
                        text="data & statistics",font="Arial 13",fill=data.black)
    canvas.create_rectangle(data.width//2+margin,startHeight2,
                            data.width-margin,
                            startHeight2+boxHeight//2-margin,fill=data.black,width=0)
    canvas.create_text(3*data.width//4,startHeight2+boxHeight//4-margin//2,
                        text="credits",font="Arial 13",fill=data.white)

#######
# Matrices functions!!!
#######

def matricesMousePressed(event,data):
    if inHomeBox(event.x,event.y,data):
        init(data)
    if data.matrixMode == "home":
        for i in range(len(data.matricesOptionsList)):
            if inIthBox(event.x,event.y,i,data):
                if i == 0: data.matrixMode = "matrixMult"
                if i == 1: data.matrixMode = "gauss"
                if i == 2: data.matrixMode = "det"
                if i == 3: data.matrixMode = "inv"
                if i == 4: data.matrixMode = "solveSystem"
                if i == 5: data.matrixMode = "LU"

def inHomeBox(x,y,data):
    if x > 0 and x < data.width//8 and y > 0 and y < data.height//10:
        return True
    return False

def inIthBox(x,y,i,data):
    width = 75
    if x > data.margin and x < data.width-data.margin:
        if y > data.height//8+data.margin+i*width:
            if y < data.height//8+data.margin+(i+1)*width:
                return True
    return False

def matrixMultKey(event,data): # A 99 line function?? What even!
    if data.tracker < data.Arows * data.Acols:
        data.matrix = "A"
    else:
        data.matrix = "B"
    print(data.matrix)
    if event.keysym == "BackSpace":
        if data.matrix == "A":
            try:                
                data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] = int(str(data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols])[:-1])
            except:
                data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] = 0
                data.point = False
                data.count = 0
                data.neg = False
        elif data.matrix == "B":
            const = data.Acols*data.Arows
            print(data.tracker-const)
            try:                
                data.matrixBlist[(data.tracker-const)//data.Bcols][(data.tracker-const)%data.Bcols] = int(str(data.matrixBlist[(data.tracker-const)//data.Bcols][(data.tracker-const)%data.Bcols])[:-1])
            except:
                data.matrixBlist[(data.tracker-const)//data.Bcols][(data.tracker-const)%data.Bcols] = 0
                data.point = False
                data.count = 0
                data.neg = False
    if event.keysym in "12345":
        if data.Arows == 0:
            data.Arows = int(event.keysym)
        elif data.Acols == 0:
            data.Acols = int(event.keysym)
        elif data.Bcols == 0:
            data.Bcols = int(event.keysym)
            data.matrixAlist = make2dList(data.Arows,data.Acols)
            data.matrixBlist = make2dList(data.Acols,data.Bcols)
            return
    if data.Bcols != 0:
        if event.char == "-": data.neg = True
        if event.keysym  == "period": data.point = True
        if event.keysym in ["Up","Down","Left","Right"]:
            if event.keysym == "Up":
                if data.matrix == "A":
                    data.tracker -= data.Acols
                elif data.matrix == "B":
                    data.tracker -= data.Bcols
            if event.keysym == "Down":
                if data.matrix == "A":
                    data.tracker += data.Acols
                elif data.matrix == "B":
                    data.tracker += data.Bcols
            if event.keysym == "Right":
                data.tracker += 1
            if event.keysym == "Left":
                data.tracker -= 1
            if data.tracker < 0 or data.tracker > data.Arows*data.Acols + data.Acols*data.Bcols:
                data.tracker %= data.Acols
            data.count = 0
            data.point = False
        if event.keysym in "0123456789":
            if data.point == False and data.neg == False:
                if data.tracker < data.Arows*data.Acols:
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] *= 10
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] += int(event.keysym)
                elif data.tracker < data.Arows*data.Acols + data.Acols*data.Bcols:
                    num = data.Arows*data.Acols
                    data.matrixBlist[(data.tracker-num)//data.Bcols][(data.tracker-num)%data.Bcols] *= 10
                    data.matrixBlist[(data.tracker-num)//data.Bcols][(data.tracker-num)%data.Bcols] += int(event.keysym)
            if data.point == True and data.neg == False:
                data.count += 1
                if data.tracker < data.Arows*data.Acols:
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] += int(event.keysym)*10**(-data.count)
                elif data.tracker < data.Arows*data.Acols + data.Acols*data.Bcols:
                    num = data.Arows*data.Acols
                    data.matrixBlist[(data.tracker-num)//data.Bcols][(data.tracker-num)%data.Bcols] += int(event.keysym)*10**(-data.count)
                    data.matrixBlist[(data.tracker-num)//data.Bcols][(data.tracker-num)%data.Bcols] = round(data.matrixBlist[(data.tracker-num)//data.Bcols][(data.tracker-num)%data.Bcols],data.count)
            if data.point == False and data.neg == True:
                if data.tracker < data.Arows*data.Acols:
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] *= 10
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] -= int(event.keysym)
                elif data.tracker < data.Arows*data.Acols + data.Acols*data.Bcols:
                    num = data.Arows*data.Acols
                    data.matrixBlist[(data.tracker-num)//data.Bcols][(data.tracker-num)%data.Bcols] *= 10
                    data.matrixBlist[(data.tracker-num)//data.Bcols][(data.tracker-num)%data.Bcols] -= int(event.keysym)
            if data.point == True and data.neg == True:
                data.count += 1
                if data.tracker < data.Arows*data.Acols:
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] -= int(event.keysym)*10**(-data.count)
                elif data.tracker < data.Arows*data.Acols + data.Acols*data.Bcols:
                    num = data.Arows*data.Acols
                    data.matrixBlist[(data.tracker-num)//data.Bcols][(data.tracker-num)%data.Bcols] -= int(event.keysym)*10**(-data.count)
                    data.matrixBlist[(data.tracker-num)//data.Bcols][(data.tracker-num)%data.Bcols] = round(data.matrixBlist[(data.tracker-num)//data.Bcols][(data.tracker-num)%data.Bcols],data.count)
    if event.keysym == "Return":
        data.tracker += 1
        data.count = 0
        data.point = False
        data.neg = False
    if event.keysym == "c":
        data.result = matrixMultiplier(data.matrixAlist,data.matrixBlist)
        for row in range(len(data.result)):
            for col in range(len(data.result[0])):
                data.result[row][col] = round(data.result[row][col],3)


def detKey(event,data):
    if event.keysym == "period": data.point = True
    if event.char == "-": data.neg = True
    if event.keysym == "BackSpace":
        try:
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] = int(str(data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim])[:-1])
        except:
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] = 0
            data.point = False
            data.count = 0
            data.neg = False
    if event.keysym in ["Up","Down","Left","Right"]:
        if event.keysym == "Up":
            data.tracker -= data.dim
        if event.keysym == "Down":
            data.tracker += data.dim
        if event.keysym == "Right":
            data.tracker += 1
        if event.keysym == "Left":
            data.tracker -= 1
        data.neg = False
        data.point = False
        data.count = 0
    if data.dim == 0 and event.keysym in "123456789":
        data.dim = int(event.keysym)
        data.matrixAlist = make2dList(data.dim,data.dim)
    elif event.keysym in "0123456789" and data.tracker < data.dim**2:
        if data.point == False and data.neg == False:
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] *= 10
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] += int(event.keysym)
        if data.point == True and data.neg == False:
            data.count += 1
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] += int(event.keysym) * 10**(-data.count)
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] = round(data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim],data.count+1)
        if data.point == False and data.neg == True:
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] *= 10
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] -= int(event.keysym)
        if data.point == True and data.neg == True:
            data.count += 1
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] -= int(event.keysym) * 10**(-data.count)
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] = round(data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim],data.count+1,3)
    elif event.keysym == "Return":
        data.neg = False
        data.tracker += 1
        data.point = False
        data.count = 0
    elif event.keysym == "c":
        data.det = det(data.matrixAlist)

def invKey(event,data):
    if event.char == "-": data.neg = True
    if event.char == ".": data.point = True
    if event.keysym == "BackSpace":
        try:          
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] = int(str(data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim])[:-1])
        except:
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] = 0
        data.point = False
        data.count = 0
        data.neg = False
    if event.keysym in ["Up","Down","Left","Right"]:
        if event.keysym == "Up":
            data.tracker -= data.dim
        if event.keysym == "Down":
            data.tracker += data.dim
        if event.keysym == "Left":
            data.tracker -= 1
        if event.keysym == "Right":
            data.tracker += 1
        data.count = 0
        data.point = False
        data.neg = False
        if data.tracker > data.dim**2 or data.tracker < 0:
            data.tracker %= data.dim**2
    if data.dim == 0 and event.keysym in "123456789":
        data.dim = int(event.keysym)
        data.matrixAlist = make2dList(data.dim,data.dim)
        data.result = make2dList(data.dim,data.dim)
    elif event.keysym in "0123456789" and data.tracker < data.dim**2:
        if data.neg == False and data.point == False:
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] *= 10
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] += int(event.keysym)
        if data.neg == False and data.point == True: 
            data.count += 1
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] += int(event.keysym) * 10 ** (-data.count)
        if data.neg == True and data.point == False:
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] *= 10
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] -= int(event.keysym)
        if data.neg == True and data.point == True:
            data.count += 1
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] -= int(event.keysym) * 10 ** (-data.count)
        if data.count > 0:
            data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim] = round(data.matrixAlist[data.tracker//data.dim][data.tracker%data.dim],data.count)
    if event.keysym == "Return":
        data.neg = False
        data.point = False
        data.tracker += 1
        data.count = 0
    elif event.keysym == "c":
        data.result = inverse(data.matrixAlist)
        for row in range(len(data.result)):
            for col in range(len(data.result[0])):
                try:
                    data.result[row][col] = round(data.result[row][col],3)
                    if abs(data.result[row][col]) < 10**(-4):
                        data.result[row][col] = 0
                except:
                    pass

def solveSystemKey(event,data): # another 99-line function, LOL
    if event.keysym == "period": data.point = True
    if event.char == "-": data.neg = True
    if data.tracker >= data.Acols*data.Arows-1:
        data.matrix = "B"
    elif data.tracker < data.Acols*data.Arows-1: 
        data.matrix = "A"
    if event.keysym == "BackSpace":
        if data.matrix == "A":
            try:                
                data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] = int(str(data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols])[:-1])
            except:
                data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] = 0
                data.point = False
                data.count = 0
                data.neg = False
        elif data.matrix == "B":
            const = data.Acols*data.Arows
            print(data.tracker-const)
            try:                
                data.targetVector[(data.tracker-const)] = int(str(data.targetVector[(data.tracker-const)])[:-1])
            except:
                data.targetVector[(data.tracker-const)] = 0
                data.point = False
                data.count = 0
                data.neg = False
    if event.keysym in ["Up","Down","Left","Right"]:
        if event.keysym == "Up":
            if data.matrix == "A":
                data.tracker -= data.Acols
            elif data.matrix == "B":
                data.tracker -= 1
        if event.keysym == "Down":
            if data.matrix == "A":
                data.tracker += data.Acols
            elif data.matrix == "B":
                data.tracker += 1
        if event.keysym == "Right":
            data.tracker += 1
        if event.keysym == "Left":
            data.tracker -= 1
        if data.tracker < 0 or data.tracker > data.Arows*data.Acols + data.Acols*data.Bcols + data.Arows:
            data.tracker %= data.Acols
        data.count = 0
        data.point = False
        data.neg = False
    if event.keysym in "12345" and data.dim == 0:
        if data.Arows == 0:
            data.Arows = int(event.keysym)
        elif data.Acols == 0:
            data.Acols = int(event.keysym)
            data.dim = 1
            return
    if data.matrixAlist == [] and data.Acols != 0: 
        data.matrixAlist = make2dList(data.Arows,data.Acols)
    if data.dim == 1:
        if data.targetVector == []:
            data.targetVector = [0] * data.Arows
        if event.keysym in "0123456789":
            if data.tracker < data.Arows*data.Acols:
                if data.point == False and data.neg == False:
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] *= 10
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] += int(event.keysym)
                elif data.point == False and data.neg == True:
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] *= 10
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] -= int(event.keysym)
                elif data.point == True and data.neg == False:
                    data.count += 1
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] += int(event.keysym)*10**(-data.count)
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] = round(data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols],3)
                elif data.point == True and data.neg == True:
                    data.count += 1
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] -= int(event.keysym)*10**(-data.count)
                    data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols] = round(data.matrixAlist[data.tracker//data.Acols][data.tracker%data.Acols],3)
            elif data.tracker < data.Arows*data.Acols + data.Arows:
                num = data.Arows*data.Acols
                if data.point == False and data.neg == False:
                    data.targetVector[data.tracker-num] *= 10
                    data.targetVector[data.tracker-num] += int(event.keysym)
                if data.point == False and data.neg == True:
                    data.targetVector[data.tracker-num] *= 10
                    data.targetVector[data.tracker-num] -= int(event.keysym)
                if data.point == True and data.neg == False:
                    data.count += 1
                    data.targetVector[data.tracker-num] += int(event.keysym)*10**(-data.count)
                    #data.targetVector[data.tracker-num] = round(data.targetVector[data.tracker-num],3)
                if data.point == True and data.neg == True:
                    data.count += 1
                    data.targetVector[data.tracker-num] -= int(event.keysym)*10**(-data.count)
                    #data.targetVector[data.tracker-num] = round(data.targetVector[data.tracker-num],3)
        if event.keysym == "Return":
            data.neg = False
            data.point = False
            data.tracker += 1
            data.count = 0
        if event.keysym == "c":
            if data.Arows == data.Acols:
                data.solution = systemsForSquareMatrices(data.matrixAlist,data.targetVector)
                for i in range(len(data.solution)):
                    data.solution[i] = round(data.solution[i],3)

def gaussKey(event,data):
    if event.keysym == "BackSpace":
        try:                
            data.matrixAlist[data.count//data.Acols][data.count%data.Acols] = int(str(data.matrixAlist[data.count//data.Acols][data.count%data.Acols])[:-1])
        except:
            data.matrixAlist[data.count//data.Acols][data.count%data.Acols] = 0
            data.point = False
            data.tracker = 0
            data.neg = False
    if event.keysym in "12345":
        if data.Arows == 0: data.Arows = int(event.keysym) 
        elif data.Acols == 0: data.Acols = int(event.keysym)
    if data.Acols != 0 and event.keysym == "Return" and data.dim != 1:
        data.matrixAlist = make2dList(data.Arows,data.Acols)
        data.dim = 1
        return
    if data.dim == 1 and data.count < data.Arows * data.Acols:
        if event.keysym == "Up":
            data.count -= data.Acols
        if event.keysym == "Down":
            data.count += data.Acols
        if event.keysym == "Right":
            data.count += 1
        if event.keysym == "Left":
            data.count -= 1
        if data.count >= data.Arows*data.Acols or data.count < 0:
            data.count %= data.Arows*data.Acols
        if event.char == "-": data.neg = True
        if event.keysym in "1234567890":
            if data.point == False and data.neg == False:
                data.matrixAlist[data.count//data.Acols][data.count%data.Acols] *= 10
                data.matrixAlist[data.count//data.Acols][data.count%data.Acols] += int(event.keysym)
            elif data.point == True and data.neg == False:
                data.tracker += 1
                data.matrixAlist[data.count//data.Acols][data.count%data.Acols] += int(event.keysym) * 10 ** (-data.tracker)
            elif data.point == False and data.neg == True:
                data.matrixAlist[data.count//data.Acols][data.count%data.Acols] *= 10
                data.matrixAlist[data.count//data.Acols][data.count%data.Acols] -= int(event.keysym)
            elif data.point == True and data.neg == True:
                data.tracker += 1
                data.matrixAlist[data.count//data.Acols][data.count%data.Acols] -= int(event.keysym) * 10 ** (-data.tracker)
        elif event.keysym == "period": data.point = True
        elif event.keysym == "Return": 
            data.neg = False
            data.count += 1
            data.point = False
            data.tracker = 0
        if event.keysym in ["Up","Down","Left","Right"]:
            data.tracker = 0
            data.neg = False
            data.point = False
    if event.keysym == "c": # compute!!!
        data.matrixBlist = gaussianElimination(data.matrixAlist)

def LUKey(event,data):
    if event.char == "-": data.neg = True
    if event.keysym in ["Up","Down","Left","Right"]:
        if event.keysym == "Up":
            data.count -= data.Acols
        if event.keysym == "Down":
            data.count += data.Acols
        if event.keysym == "Left":
            data.count -= 1
        if event.keysym == "Right":
            data.count += 1
        data.neg = False
        data.point = False
        data.tracker = 0
    if event.keysym == "BackSpace":
        try:
            data.matrixAlist[data.count//data.Acols][data.count%data.Acols] = int(str(data.matrixAlist[data.count//data.Acols][data.count%data.Acols])[:-1])
        except:
            data.matrixAlist[data.count//data.Acols][data.count%data.Acols] = 0
        data.neg = False
        data.point = False
        data.tracker = 0
    if event.keysym in "12345":
        if data.Arows == 0: data.Arows = int(event.keysym) 
        elif data.Acols == 0: data.Acols = int(event.keysym)
    if data.Acols != 0 and event.keysym == "Return" and data.dim != 1:
        data.matrixAlist = make2dList(data.Arows,data.Acols)
        data.dim = 1
        return
    if data.dim == 1 and data.count < data.Arows * data.Acols:
        if event.keysym in "1234567890":
            if data.point == False and data.neg == False:
                data.matrixAlist[data.count//data.Acols][data.count%data.Acols] *= 10
                data.matrixAlist[data.count//data.Acols][data.count%data.Acols] += int(event.keysym)
            elif data.point == True and data.neg == False:
                data.tracker += 1
                data.matrixAlist[data.count//data.Acols][data.count%data.Acols] += int(event.keysym) * 10 ** (-data.tracker)
            elif data.point == False and data.neg == True:
                data.matrixAlist[data.count//data.Acols][data.count%data.Acols] *= 10
                data.matrixAlist[data.count//data.Acols][data.count%data.Acols] -= int(event.keysym)
            elif data.point == True and data.neg == True:
                data.tracker += 1
                data.matrixAlist[data.count//data.Acols][data.count%data.Acols] -= int(event.keysym) * 10 ** (-data.tracker)
        elif event.keysym == "period": data.point = True
        elif event.keysym == "Return": 
            data.neg = False
            data.count += 1
            data.point = False
            data.tracker = 0
    if event.keysym == "c":
        data.det = 1
        data.result = gaussianElimination(data.matrixAlist) # This is our U
        data.matrixBlist = matrixMultiplier(data.matrixAlist,inverse(gaussianElimination(data.matrixAlist)))
        for row in range(len(data.result)):
            for col in range(len(data.result[0])):
                data.result[row][col] = round(data.result[row][col],2)
                if abs(data.result[row][col]) < 0.01:
                    data.result[row][col] = float(0)
        for row in range(len(data.matrixBlist)):
            for col in range(len(data.matrixBlist[0])):
                data.matrixBlist[row][col] = round(data.matrixBlist[row][col],2)
                if abs(data.matrixBlist[row][col]) < 0.01:
                    data.matrixBlist[row][col] = float(0)

def matricesKeyPressed(event,data):
    if data.matrixMode == "matrixMult":
        matrixMultKey(event,data)
    if data.matrixMode == "det":
        detKey(event,data)
    if data.matrixMode == "inv":
        invKey(event,data)
    if data.matrixMode == "solveSystem":
        solveSystemKey(event,data)
    if data.matrixMode == "gauss":
        gaussKey(event,data)
    if data.matrixMode == "LU":
        LUKey(event,data)

def matricesTimer(data):
    pass

def drawBoxes(canvas,data):
    unit = 35
    canvas.create_line(data.width//6,data.height//5+3*data.margin,
                    data.width//6,data.height//5+5*unit)
    canvas.create_line(data.width//2-data.margin,data.height//5+3*data.margin,
                       data.width//2-data.margin,data.height//5+5*unit)
    canvas.create_line(5*data.width//6,data.height//5+3*data.margin,
            5*data.width//6,data.height//5+5*unit)
    canvas.create_line(data.width//2+data.margin,data.height//5+3*data.margin,
            data.width//2+data.margin,data.height//5+5*unit)
    shiftx,shifty = 90,180
    canvas.create_line(data.width//6+shiftx,data.height//5+3*data.margin+shifty,
                    data.width//6+shiftx,data.height//5+5*unit+shifty)
    canvas.create_line(data.width//2-data.margin+shiftx,
                       data.height//5+3*data.margin+shifty,
                       data.width//2-data.margin+shiftx,data.height//5+5*unit+shifty)

def drawEntries(canvas,data):
    for row in range(data.Arows):
        for col in range(data.Acols):
            dx = (data.width//3-data.margin)//data.Acols
            dy = (data.width//3-data.margin)//data.Arows
            startx,starty = data.width//6,data.height//5+3*data.margin
            canvas.create_text(startx+(col+0.5)*dx,starty+(row+0.5)*dy,
                               text=str(data.matrixAlist[row][col]))
            if data.tracker == row*data.Acols + col:
                canvas.create_rectangle(startx+(col+0.5)*dx-15,starty+(row+0.5)*dy-15,
                    startx+(col+0.5)*dx+15,starty+(row+0.5)*dy+15,outline=data.blue)
    for row in range(data.Acols):
        for col in range(data.Bcols):
            dx = (data.width//3-data.margin)//data.Bcols
            dy = (data.width//3-data.margin)//data.Acols
            startx,starty = data.width//2+data.margin,data.height//5+3*data.margin
            canvas.create_text(startx+(col+0.5)*dx,starty+(row+0.5)*dy,
                               text=str(data.matrixBlist[row][col]))
            if data.tracker == data.Arows*data.Acols + row*data.Bcols + col:
                canvas.create_rectangle(startx+(col+0.5)*dx-15,starty+(row+0.5)*dy-15,
                    startx+(col+0.5)*dx+15,starty+(row+0.5)*dy+15,outline=data.blue)
    shiftx,shifty = 90,180
    # quick editing the entries so they're presentable!
    if data.Bcols > 3 and data.result != []:
        for row in range(data.Arows):
            for col in range(data.Bcols):
                data.result[row][col] = round(data.result[row][col],1)
    if data.result != []:
        for row in range(data.Arows):
            for col in range(data.Bcols):
                dx = (data.width//3-data.margin)//data.Bcols
                dy = (data.width//3-data.margin)//data.Arows
                startx,starty = data.width//6+shiftx,data.height//5+3*data.margin+shifty
                canvas.create_text(startx+(col+0.5)*dx,starty+(row+0.5)*dy,
                               text=str(data.result[row][col]))

def drawSquareMatrixEntries(canvas,data,x,y,A,n=False):
    dim = data.dim
    if data.dim == 0: return
    dx = (data.width//2-x)*2//data.dim
    for i in range(dim):
        for j in range(dim):
            if data.tracker == i*data.dim + j and (data.matrixMode == "det" or n == True):
                canvas.create_rectangle(x+(j+0.5)*dx-15,y+2*(i+0.5)*dx//3-15,x+(j+0.5)*dx+15,y+2*(i+0.5)*dx//3+15,outline=data.blue)
            canvas.create_text(x+(j+0.5)*dx,y+2*(i+0.5)*dx//3,
            text=A[i][j],font="Arial",fill=data.black)

def drawDetScreen(canvas,data):
    canvas.create_text(data.width//2,data.height//8,
    text="T a k e   t h e   d e t e r m i n a n t   o f   a   m a t r i x",font="Arial 20 bold",fill=data.black)
    canvas.create_text(data.width//2,data.height//8+3*data.margin,
    text="dimension: "+str(data.dim),font="Arial",fill=data.black)
    canvas.create_text(data.width//2,data.height-3*data.margin,
    text="type numbers to toggle dimension and entries; use arrows to navigate, c to compute",font="Arial",fill=data.black)
    canvas.create_line(data.width//2-10*data.margin*(data.dim//2),
    data.startHeight, data.width//2-10*data.margin*(data.dim//2),
    data.startHeight+5*data.margin*data.dim)
    canvas.create_line(data.width//2+10*data.margin*(data.dim//2),
    data.startHeight, data.width//2+10*data.margin*(data.dim//2),
    data.startHeight+5*data.margin*data.dim)
    canvas.create_text(data.width//2,data.height-16*data.margin+25,
    text="result = "+str(data.det),font="Arial",fill=data.black)
    drawSquareMatrixEntries(canvas,data,
    data.width//2-10*data.margin*(data.dim//2),data.startHeight,
    data.matrixAlist)

def drawMatricesHomeScreen(canvas,data):
    canvas.create_text(data.width//2,data.height//9,
                        text="M a t r i c e s  C o m p u t a t i o n s",
                        font="Arial 20 bold",fill=data.black)
    width = 75
    for i in range(len(data.matricesOptionsList)-2):
        if i % 2 == 0: color,textcolor = data.blue,data.white
        else: color,textcolor = data.yellow,data.black
        canvas.create_rectangle(data.margin,
                                data.height//8+data.margin+i*width,
                                data.width-data.margin,
                                data.height//8+data.margin+(i+1)*width,
                                fill = color,width=0)
        canvas.create_text(data.width//2,
                        data.height//8+data.margin+(i+0.5)*width,
                        text=data.matricesOptionsList[i],fill=textcolor)

def drawMatrixMultScreen(canvas,data):
    canvas.create_text(data.width//2,data.height//8,
                        text="M u l t i p l y   t w o   M a t r i c e s",font="Arial 20 bold",fill=data.black)
    canvas.create_text(data.width//2,data.height-2*data.margin,
                        text="type numbers to pick dimensions and entries; use arrows to navigate, c to compute",font="Arial",fill=data.black)
    canvas.create_text(data.width//2-data.margin,data.height//5,text="A      B",
                        font="Arial 30 bold",fill=data.black)
    canvas.create_text(data.width//2-data.margin,data.height//5+2*data.margin,
                        text="x               x",fill=data.black)
    canvas.create_text(data.width//2-data.margin*4.5,data.height//5+2*data.margin,
                        text=str(data.Arows) + "    " + str(data.Acols),fill=data.black)
    canvas.create_text(data.width//2+data.margin*2.5,data.height//5+2*data.margin,
                        text=str(data.Acols) + "   " + str(data.Bcols),fill=data.black)
    if data.Bcols != 0:
        drawBoxes(canvas,data)
        drawEntries(canvas,data)

def drawInvMatrixScreen(canvas,data):
    canvas.create_text(data.width//2,data.height//8,
    text="Find the Inverse of a Matrix",font="Arial 20 bold",fill=data.black)
    canvas.create_text(data.width//2,data.height//8+3*data.margin,
    text="dimension: "+str(data.dim),font="Arial",fill=data.black)
    canvas.create_text(data.width//2,data.height-3*data.margin,
    text="type numbers to toggle dimension and entries; use arrows to navigate, c to compute",font="Arial",fill=data.black)
    canvas.create_text(data.width//6,3*data.height//4,text="result = "+data.msg,
        font="Arial",fill=data.black,anchor=W)
    canvas.create_line(data.width//2-6*data.margin*(data.dim//2),
    data.startHeight, data.width//2-6*data.margin*(data.dim//2),
    data.startHeight+3*data.margin*data.dim)
    canvas.create_line(data.width//2+6*data.margin*(data.dim//2),
    data.startHeight, data.width//2+6*data.margin*(data.dim//2),
    data.startHeight+3*data.margin*data.dim)
    if data.dim != 0:
        canvas.create_line(data.width//2-6*data.margin*(data.dim//2),
        data.startHeight+200, data.width//2-6*data.margin*(data.dim//2),
        data.startHeight+3*data.margin*data.dim+200)
        canvas.create_line(data.width//2+6*data.margin*(data.dim//2),
        data.startHeight+200, data.width//2+6*data.margin*(data.dim//2),
        data.startHeight+3*data.margin*data.dim+200)
    drawSquareMatrixEntries(canvas,data,
    data.width//2-6*data.margin*(data.dim//2),data.startHeight,data.matrixAlist,n=True) 
    if isinstance(data.result,list):       
        drawSquareMatrixEntries(canvas,data,
        data.width//2-6*data.margin*(data.dim//2),3*data.width//4-100,data.result)
    else: 
        data.msg = "this matrix is NOT invertible"

def drawSolveSystemScreen(canvas,data):
    canvas.create_text(data.width//2,data.height-2*data.margin,
        text="type numbers to toggle dimensions and entries; use arrows to navigate, c to compute",
        font="Arial",fill=data.black)
    canvas.create_text(data.width//2,data.height//14,text="L i n e a r  S y s t e m s",
    font="Arial 20 bold",fill=data.black)
    canvas.create_text(data.width//2,data.height//14+2*data.margin,
    text="dimensions for A: " + str(data.Arows) + " x " + str(data.Acols),font="Arial",fill=data.black)
    if data.Acols != 0:
        size = 50
        canvas.create_line(70,80,70,80+data.Arows*size)
        canvas.create_line(70+data.Acols*size,80,
        70+data.Acols*size,80+data.Arows*size)
        canvas.create_line(70+data.Acols*size+data.margin,80,
        70+data.Acols*size+data.margin,80+data.Acols*size)
        canvas.create_line(70+data.Acols*size+data.margin+size,80,
        70+data.Acols*size+data.margin+size,80+data.Acols*size)
        for i in range(data.Acols):
            canvas.create_text(70+(data.Acols+0.5)*size+data.margin,
            80+(i+0.5)*size,text="X%d" % (i+1))
        canvas.create_text(70+data.Acols*size+3*data.margin+size,
        80+data.Arows*size//2,text=" = ",font="Arial 30",fill=data.black)
        canvas.create_line(70+data.Acols*size+3*data.margin+2*size,
        80,70+data.Acols*size+3*data.margin+2*size,80+data.Arows*size)
        canvas.create_line(70+data.Acols*size+3*data.margin+3*size,
        80,70+data.Acols*size+3*data.margin+3*size,80+data.Arows*size)
        for row in range(len(data.matrixAlist)):
            for col in range(len(data.matrixAlist[0])):
                if data.tracker == row*data.Acols + col:
                    canvas.create_rectangle(70+(col+0.5)*size-15,80+(row+0.5)*size-15,70+(col+0.5)*size+15,80+(row+0.5)*size+15,outline=data.blue)
                canvas.create_text(70+(col+0.5)*size,80+(row+0.5)*size,
                text=data.matrixAlist[row][col],font="Arial",fill=data.black)
            if data.tracker == data.Arows*data.Acols + row:
                canvas.create_rectangle(70+data.Acols*size+3*data.margin+2.5*size-15,
                    80+(row+0.5)*size-15,70+data.Acols*size+3*data.margin+2.5*size+15,
                    80+(row+0.5)*size+15,outline=data.blue)
            canvas.create_text(70+data.Acols*size+3*data.margin+2.5*size,
            80+(row+0.5)*size,text=data.targetVector[row],font="Arial",fill=data.black)
        canvas.create_text(data.width//5,3*data.height//4,
        text="Solution:",font="Arial",fill=data.black)
        shiftx,shifty = -175,225
        for i in range(data.Acols):
            canvas.create_text(70+data.Acols*size+3*data.margin+2.5*size+shiftx,
                80+shifty+(i+0.5)*data.Arows*size/data.Acols,text="X%d" % (i+1),fill=data.black,font="Arial")
            canvas.create_text(70+data.Acols*size+3*data.margin+2.5*size+shiftx+50,
                80+shifty+(data.Acols/2)*data.Arows*size/data.Acols,text="=",font="Arial 30",fill=data.black)
        canvas.create_line(70+data.Acols*size+3*data.margin+2*size+shiftx,
        80+shifty,70+data.Acols*size+3*data.margin+2*size+shiftx,80+data.Arows*size+shifty,fill=data.black)
        canvas.create_line(70+data.Acols*size+3*data.margin+3*size+shiftx,
        80+shifty,70+data.Acols*size+3*data.margin+3*size+shiftx,80+data.Arows*size+shifty,fill=data.black)
        if not isinstance(data.solution,str):
            for j in range(len(data.solution)):
                canvas.create_text(170+data.Acols*size+3*data.margin+2.5*size+shiftx,
                    80+shifty+(j+0.5)*data.Arows*size/data.Acols,text=str(data.solution[j]),fill=data.black,font="Arial")
        canvas.create_line(170+data.Acols*size+3*data.margin+2*size+shiftx,
        80+shifty,170+data.Acols*size+3*data.margin+2*size+shiftx,80+data.Arows*size+shifty,fill=data.black)
        canvas.create_line(170+data.Acols*size+3*data.margin+3*size+shiftx,
        80+shifty,170+data.Acols*size+3*data.margin+3*size+shiftx,80+data.Arows*size+shifty,fill=data.black)

def drawGauss(canvas,data):
    canvas.create_text(data.width//2,data.height-data.margin*2,
        text="type numbers for dimensions and entries; hit enter to create your matrix, c to compute",
        font="Arial",fill=data.black)
    canvas.create_text(data.width//2,data.height//10-20,text="R o w - R e d u c e  a  M a t r i x",
        font="Arial 20 bold",fill=data.black)
    canvas.create_text(data.width//2,data.height//8-data.margin,
        text="dimensions: %d rows, %d columns" % (data.Arows,data.Acols), font="Arial",fill=data.black)
    halfMatrixSize = 100
    canvas.create_line(data.width//2-halfMatrixSize,data.height//3-halfMatrixSize,
        data.width//2-halfMatrixSize,data.height//3+halfMatrixSize,fill=data.black)
    canvas.create_line(data.width//2+halfMatrixSize,data.height//3-halfMatrixSize,
        data.width//2+halfMatrixSize,data.height//3+halfMatrixSize,fill=data.black)
    if data.Acols != 0 and data.Arows != 0:
        colsize = 2*halfMatrixSize / data.Acols
        rowsize = 2*halfMatrixSize / data.Arows
    if data.dim == 1:
        for row in range(data.Arows):
            for col in range(data.Acols):
                if data.count == row*data.Acols + col:
                    canvas.create_rectangle(data.width//2-halfMatrixSize+(col+0.5)*colsize-15,
                        data.height//3-halfMatrixSize+(row+0.5)*rowsize-15,data.width//2-halfMatrixSize+(col+0.5)*colsize+15,
                        data.height//3-halfMatrixSize+(row+0.5)*rowsize+15,outline=data.blue)
                canvas.create_text(data.width//2-halfMatrixSize+(col+0.5)*colsize,
                    data.height//3-halfMatrixSize+(row+0.5)*rowsize,
                    text=str(data.matrixAlist[row][col]),font="Arial",fill=data.black)
    canvas.create_line(data.width//2-halfMatrixSize,
        2*data.height//3-halfMatrixSize+5*data.margin,
        data.width//2-halfMatrixSize,
        2*data.height//3+halfMatrixSize+5*data.margin,fill=data.black)
    canvas.create_line(data.width//2+halfMatrixSize,
        2*data.height//3-halfMatrixSize+5*data.margin,
        data.width//2+halfMatrixSize,
        2*data.height//3+halfMatrixSize+5*data.margin,fill=data.black)
    if data.matrixBlist != []:
        for row in range(data.Arows):
            for col in range(data.Acols):
                canvas.create_text(data.width//2-halfMatrixSize+(col+0.5)*colsize,
                    2*data.height//3-halfMatrixSize+(row+0.5)*rowsize+5*data.margin,
                    text=str(data.matrixBlist[row][col]),font="Arial",fill=data.black)

def drawLU(canvas,data):
    canvas.create_text(data.width//2,data.height-2*data.margin,text="type numbers for dimensions and entries; hit enter to create your matrix, c to compute",fill=data.black,font="Arial")
    canvas.create_text(data.width//2,data.height//10-2*data.margin,
        text="F i n d  L U  D e c o m p o s i t i o n",font="Arial 20 bold",fill=data.black)
    canvas.create_text(data.width//2,data.height//8-data.margin,
        text="dimensions: %d rows, %d columns" % (data.Arows,data.Acols),
        font="Arial",fill=data.black)
    halfMatrixSize = 100
    canvas.create_text(data.width//5,data.height//3,text="A =",fill=data.black,font="Arial 25")
    canvas.create_text(30,3*data.height//4,text="L",font="Arial 25",fill=data.black)
    canvas.create_text(data.width-30,3*data.height//4,text="U",font="Arial 25",fill=data.black)
    canvas.create_line(data.width//2-halfMatrixSize,data.height//3-halfMatrixSize,
        data.width//2-halfMatrixSize,data.height//3+halfMatrixSize,fill=data.black)
    canvas.create_line(data.width//2+halfMatrixSize,data.height//3-halfMatrixSize,
        data.width//2+halfMatrixSize,data.height//3+halfMatrixSize,fill=data.black)
    if data.Acols != 0 and data.Arows != 0:
        colsize = 2*halfMatrixSize / data.Acols
        rowsize = 2*halfMatrixSize / data.Arows
    shiftx,shifty = 100,215
    canvas.create_line(data.width//2-halfMatrixSize-shiftx-data.margin,
        data.height//3-halfMatrixSize+shifty,
        data.width//2-halfMatrixSize-shiftx-data.margin,
        data.height//3+halfMatrixSize+shifty,fill=data.black)
    canvas.create_line(data.width//2+halfMatrixSize-shiftx-data.margin,
        data.height//3-halfMatrixSize+shifty,
        data.width//2+halfMatrixSize-shiftx-data.margin,
        data.height//3+halfMatrixSize+shifty,fill=data.black) # These lines are for L
    canvas.create_line(data.width//2-halfMatrixSize+shiftx+data.margin,
        data.height//3-halfMatrixSize+shifty,
        data.width//2-halfMatrixSize+shiftx+data.margin,
        data.height//3+halfMatrixSize+shifty,fill=data.black)
    canvas.create_line(data.width//2+halfMatrixSize+shiftx+data.margin,
        data.height//3-halfMatrixSize+shifty,
        data.width//2+halfMatrixSize+shiftx+data.margin,
        data.height//3+halfMatrixSize+shifty,fill=data.black) # These lines are for U
    if data.dim == 1:
        for row in range(data.Arows):
            for col in range(data.Acols):
                if data.count == row*data.Acols + col:
                    canvas.create_rectangle(data.width//2-halfMatrixSize+(col+0.5)*colsize-15,
                    data.height//3-halfMatrixSize+(row+0.5)*rowsize-15,data.width//2-halfMatrixSize+(col+0.5)*colsize+15,
                    data.height//3-halfMatrixSize+(row+0.5)*rowsize+15,outline=data.blue)
                canvas.create_text(data.width//2-halfMatrixSize+(col+0.5)*colsize,
                    data.height//3-halfMatrixSize+(row+0.5)*rowsize,
                    text=str(data.matrixAlist[row][col]),font="Arial",fill=data.black)
    if data.det == 1:
        resultcolsize = 2*halfMatrixSize / len(data.result[0])
        resultrowsize = 2*halfMatrixSize / len(data.result)
        Bcolsize = 2*halfMatrixSize / len(data.matrixBlist[0])
        BrowSize = 2*halfMatrixSize / len(data.matrixBlist)
        for row in range(len(data.matrixBlist)):
            for col in range(len(data.matrixBlist[0])):
                canvas.create_text(data.width//2-halfMatrixSize-shiftx-data.margin+(col+0.5)*resultcolsize,
                    data.height//3-halfMatrixSize+shifty+(row+0.5)*resultrowsize,
                    text=str(data.matrixBlist[row][col]),font="Arial",fill=data.black)
        for row in range(len(data.result)):
            for col in range(len(data.result[0])):
                canvas.create_text(data.width//2-halfMatrixSize+shiftx+data.margin+(col+0.5)*Bcolsize,
                    data.height//3-halfMatrixSize+shifty+(row+0.5)*BrowSize,
                    text=str(data.result[row][col]),font="Arial",fill=data.black)

def matricesRedrawAll(canvas,data):
    canvas.create_rectangle(0,0,data.width,data.height,fill=data.white,width=0)
    canvas.create_rectangle(0,0,data.width//8,data.height//10,fill=data.peach,width=0)
    canvas.create_text(data.width//16,data.height//20,text="home",fill=data.white)
    if data.matrixMode == "home":
        drawMatricesHomeScreen(canvas,data)
    elif data.matrixMode == "matrixMult":
        drawMatrixMultScreen(canvas,data)
    elif data.matrixMode == "det":
        # determinants
        drawDetScreen(canvas,data)
    elif data.matrixMode == "inv":
        drawInvMatrixScreen(canvas,data)
    elif data.matrixMode == "solveSystem":
        drawSolveSystemScreen(canvas,data)
    elif data.matrixMode == "gauss":
        drawGauss(canvas,data)
    elif data.matrixMode == "LU":
        drawLU(canvas,data)

######
# graphing functions
######

def f(x,data,i): return eval(data.fct[i]) # I think this is the shortest function in the whole project

def graphMousePressed(event,data):
    if data.graphMode == "real":
        init(data)
        return
    toggle = 15
    if data.graphMode == None:
        if inBox(data.margin,data.startHeight,data.width//2-data.margin,
        data.startHeight*4,event.x,event.y):
            data.graphMode = "real"
            return
        elif inBox(data.width//2+data.margin,data.startHeight,data.width-data.margin,data.startHeight*4,event.x,event.y):
            data.graphMode = "r2"
            return
    elif inBox(data.width-toggle,data.height-toggle,data.width,data.height,event.x,event.y):
        if data.det == 0:
            data.det = 1
    if data.graphMode == None or data.graphMode == "r2":
        if inBox(0,0,data.width//8,data.height//10,event.x,event.y):
            init(data)

def realKey(event,data):
    if data.det == 1:
        if event.keysym in "0123456789":
            if data.dum == 0: data.xmax = int(str(data.xmax)+str(event.keysym))
            if data.dum == 1: data.ymax = int(str(data.ymax)+str(event.keysym))
            if data.dum == 2: data.xstep = int(str(data.xstep)+str(event.keysym))
            if data.dum == 3: data.ystep = int(str(data.ystep)+str(event.keysym))
            return
        elif event.keysym == "BackSpace":
            if data.dum == 0: 
                if str(data.xmax)[:-1] != "":
                    data.xmax = int(str(data.xmax)[:-1])
                else: data.xmax = ""
            if data.dum == 1: 
                if str(data.ymax)[:-1] != "":
                    data.ymax = int(str(data.ymax)[:-1])
                else: data.ymax = ""
            if data.dum == 2: 
                if str(data.xstep)[:-1] != "":
                    data.xstep = int(str(data.xstep)[:-1])
                else: data.xstep = ""
            if data.dum == 3: 
                if str(data.ystep)[:-1] != "":
                    data.ystep = int(str(data.ystep)[:-1])
                else: data.ystep = ""
    if event.char in "0123456789x.*/+-()" or event.char.isalpha() and data.det == 0:
        data.graph = False
        data.fct[data.count] += event.char
    if event.keysym == "Return":
        data.graph = True
        data.count += 1
        data.count %= 3
    if event.keysym == "BackSpace":
        data.graph = False
        data.fct[data.count] = data.fct[data.count][:-1]
    if event.keysym == "Up":
        data.count -= 1
        data.count %= 3
        data.dum -= 1
        data.dum %= 4
    if event.keysym == "Down":
        data.count += 1
        data.count %= 3
        data.dum += 1
        data.dum %= 4
    if event.keysym == "Right":
        data.det = 1
    if event.keysym == "Left":
        data.det = 0

def r2Key(event,data):
    if data.matrixAlist == []:
        data.matrixAlist = make2dList(2,2,0)
    if data.matrixAlist[-1][-1] == 0 and event.keysym in "0123456789":
        data.matrixAlist[data.count%2][data.count//2] = int(event.keysym)
        data.count += 1
    if event.keysym == "Return":
        # if data.dim == 0: data.matrixAlist = matrixTranspose(data.matrixAlist)
        data.dim = 1
        for i in range(len(data.vectorList)):
            # note: Idk why I needed to do both things backwards, but that's what worked ://///
            data.finalVectorList[i] = matrixMultiplier(data.matrixAlist,[[data.vectorList[i][1]],[data.vectorList[i][0]]])
            data.finalVectorList[i] = [data.finalVectorList[i][1][0],data.finalVectorList[i][0][0]]
        data.start = copy.deepcopy(data.vectorList)

def otherKeyPressed(event,data): # I think this is for drawing gridlines
    start1 = copy.deepcopy(data.xValuesH)
    start2 = copy.deepcopy(data.yValuesH)
    start3 = copy.deepcopy(data.xValuesV)
    start4 = copy.deepcopy(data.yValuesV)
    a = data.matrixAlist[0][0] 
    b = data.matrixAlist[0][1] 
    c = data.matrixAlist[1][0]
    d = data.matrixAlist[1][1]
    for i in range(-20,21):
        data.xValuesV[i][0] -= data.width//2
        data.yValuesV[i][0] -= data.width//2
        data.xValuesV[i][0] = a*data.xValuesV[i][0] - b*data.yValuesV[i][0]
        data.xValuesV[i][0] += data.width//2
        data.yValuesV[i][0] += data.width//2

        data.xValuesV[i][1] -= data.width//2
        data.yValuesV[i][1] -= data.width//2
        data.xValuesV[i][1] = (a*data.xValuesV[i][1] - b*data.yValuesV[i][1])
        data.xValuesV[i][1] += data.width//2
        data.yValuesV[i][1] += data.width//2

        data.yValuesH[i][0] -= data.height//2
        data.xValuesH[i][0] -= data.height//2
        data.yValuesH[i][0] = c*data.xValuesH[i][0] - d*data.yValuesH[i][0]
        data.yValuesH[i][0] += data.height//2
        data.xValuesH[i][0] += data.height//2

        data.yValuesH[i][1] -= data.height//2
        data.xValuesH[i][1] -= data.height//2
        data.yValuesH[i][1] = c*data.xValuesH[i][1] - d*data.yValuesH[i][1]
        data.yValuesH[i][1] += data.height//2
        data.xValuesH[i][1] += data.height//2

        # this is weird: I need to flip all the x values for some reason
        data.xValuesH[i][0] -= data.height//2
        data.xValuesH[i][1] -= data.height//2
        data.xValuesH[i][0] *= -1
        data.xValuesH[i][1] *= -1
        data.xValuesH[i][0] += data.height//2
        data.xValuesH[i][1] += data.height//2

        # this is also weird: I need to flip all lines across the line y=x
        data.xValuesH[i][0],data.yValuesV[i][0] = data.yValuesV[i][0],data.xValuesH[i][0]
        data.yValuesH[i][0],data.xValuesV[i][0] = data.xValuesV[i][0],data.yValuesH[i][0]
        data.xValuesH[i][1],data.yValuesV[i][1] = data.yValuesV[i][1],data.xValuesH[i][1]
        data.yValuesH[i][1],data.xValuesV[i][1] = data.xValuesV[i][1],data.yValuesH[i][1]
    data.newXvaluesH = copy.deepcopy(data.xValuesH)
    data.newYvaluesH = copy.deepcopy(data.yValuesH)
    data.newXvaluesV = copy.deepcopy(data.xValuesV)
    data.newYvaluesV = copy.deepcopy(data.yValuesV)
    data.xValuesH = start1
    data.yValuesH = start2
    data.xValuesV = start3
    data.yValuesV = start4

def graphKeyPressed(event,data):
    if data.graphMode == "real":
        realKey(event,data)
    if data.graphMode == "r2":
        r2Key(event,data)
    if data.dim == 1:
        otherKeyPressed(event,data)

def graphTimer(data):
    if data.dim == 1:
        step = 30
        for i in range(len(data.finalVectorList)):
            for j in range(2):
                if abs(data.finalVectorList[i][j] - data.vectorList[i][j]) > 0.05: 
                    data.vectorList[i][j] += (data.finalVectorList[i][j]-data.start[i][j])/step
        fastStep = 10
        for i in range(-20,21):
            if abs(data.xValuesH[i][0]-data.newXvaluesH[i][0]) > 0.05:
                data.xValuesH[i][0] += (data.newXvaluesH[i][0]-data.xValuesH[i][0])/fastStep
            if abs(data.newXvaluesH[i][1]-data.xValuesH[i][1]) > 0.05:
                data.xValuesH[i][1] += (data.newXvaluesH[i][1]-data.xValuesH[i][1])/fastStep
            if abs(data.newXvaluesV[i][0]-data.xValuesV[i][0]) > 0.05:
                data.xValuesV[i][0] += (data.newXvaluesV[i][0]-data.xValuesV[i][0])/fastStep
            if abs(data.newXvaluesV[i][1]-data.xValuesV[i][1]) > 0.05:
                data.xValuesV[i][1] += (data.newXvaluesV[i][1]-data.xValuesV[i][1])/fastStep
            if abs(data.newYvaluesV[i][0]-data.yValuesV[i][0]) > 0.05:
                data.yValuesV[i][0] += (data.newYvaluesV[i][0]-data.yValuesV[i][0])/fastStep
            if abs(data.newYvaluesV[i][1]-data.yValuesV[i][1]) > 0.05:
                data.yValuesV[i][1] += (data.newYvaluesV[i][1]-data.yValuesV[i][1])/fastStep
            if abs(data.newYvaluesH[i][0]-data.yValuesH[i][0]) > 0.05:
                data.yValuesH[i][0] += (data.newYvaluesH[i][0]-data.yValuesH[i][0])/fastStep
            if abs(data.newYvaluesH[i][1]-data.yValuesH[i][1]) > 0.05:
                data.yValuesH[i][1] += (data.newYvaluesH[i][1]-data.yValuesH[i][1])/fastStep

def drawGraphMenu(canvas,data):
    canvas.create_rectangle(data.margin,data.startHeight,data.width//2-data.margin,
    data.startHeight*4,fill=data.yellow,width=0)
    canvas.create_rectangle(data.width//2+data.margin,data.startHeight,
    data.width-data.margin,data.startHeight*4,fill=data.blue,width=0)
    canvas.create_text(data.width//4,5*data.startHeight//2,
    text="graph real functions",font="Arial",fill=data.black)
    canvas.create_text(3*data.width//4,5*data.startHeight//2,
    text="R^2 -> R^2 linear maps",font="Arial",fill=data.white)

# this function, drawRealFunctions, takes advantage of a 112 HW problem from Fall 14
# http://www.kosbie.net/cmu/fall-14/15-112/notes/hw3.html
# Of course, I produced and debugged a solution, which I 
# appropriately modified to fit into this project.

def drawRealFunctions(canvas, data): # an 87-line function, lol
    canvas.create_rectangle(0,0,data.width//4,data.height//5,fill=data.white,width=0)
    toggle = 15
    if data.det == 0:
        canvas.create_rectangle(data.width-toggle,data.height-toggle,
            data.width,data.height,fill=data.blue)
        canvas.create_line(data.width-5*toggle/6,data.height-5*toggle/6,
            data.width,data.height,fill=data.black)
        canvas.create_line(data.width-5*toggle/6,data.height-5*toggle/6,
            data.width-5*toggle/6,data.height-5*toggle/6+3,fill=data.black)
        canvas.create_line(data.width-5*toggle/6,data.height-5*toggle/6,
            data.width-5*toggle/6+3,data.height-5*toggle/6,fill=data.black)
    if data.det == 1:
        canvas.create_rectangle(data.width-7*toggle,data.height-7*toggle,
            data.width,data.height,fill=data.blue)
        i = data.dum
        canvas.create_rectangle(data.width-7*toggle+75-9,data.height-7*toggle+(1+2.5*i)*data.margin-9,
            data.width-7*toggle+75+9,data.height-7*toggle+(1+2.5*i)*data.margin+9,width=1,outline=data.white)
        canvas.create_text(data.width-7*toggle,data.height-7*toggle+data.margin,text=" xmax =",anchor=W,font="Arial",fill=data.white)
        canvas.create_text(data.width-7*toggle,data.height-7*toggle+3.5*data.margin,text=" ymax =",anchor=W,font="Arial",fill=data.white)
        canvas.create_text(data.width-7*toggle,data.height-7*toggle+6*data.margin,text=" xstep =",anchor=W,font="Arial",fill=data.white)
        canvas.create_text(data.width-7*toggle,data.height-7*toggle+8.5*data.margin,text=" ystep =",anchor=W,font="Arial",fill=data.white)
        canvas.create_text(data.width-7*toggle+75,data.height-7*toggle+data.margin,text=str(data.xmax),font="Arial",fill=data.white)
        canvas.create_text(data.width-7*toggle+75,data.height-7*toggle+3.5*data.margin,text=str(data.ymax),font="Arial",fill=data.white)
        canvas.create_text(data.width-7*toggle+75,data.height-7*toggle+6*data.margin,text=str(data.xstep),font="Arial",fill=data.white)
        canvas.create_text(data.width-7*toggle+75,data.height-7*toggle+8.5*data.margin,text=str(data.ystep),font="Arial",fill=data.white)
    canvas.create_text(data.width//8,3,text="Functions",anchor=N,font="Arial",fill=data.black)
    j = data.count 
    canvas.create_line(0,3*(j+1.4)*data.margin,40,3*(j+1.4)*data.margin,fill=data.black)
    canvas.create_text(data.margin,3*data.margin,text="f(x)="+str(data.fct[0]),anchor=W,font="Arial",fill=data.blue)
    canvas.create_text(data.margin,6*data.margin,text="g(x)="+str(data.fct[1]),anchor=W,font="Arial",fill=data.peach)
    canvas.create_text(data.margin,9*data.margin,text="h(x)="+str(data.fct[2]),anchor=W,font="Arial",fill=data.black)
    winWidth,winHeight = data.width,data.height
    (cx, cy) = (winWidth/2, winHeight/2)
    # draw axes
    canvas.create_line(0, cy, winWidth, cy,fill=data.black)
    canvas.create_line(cx, 0, cx, winHeight,fill=data.black)
    xmax = data.xmax
    ymax = data.ymax
    xstep = data.xstep
    ystep = data.ystep
    try:
        scalex = data.width / (2*xmax)
        scaley = data.height/ (2*ymax)
    except:
        pass
    mark = 5
    offset = 15
    if isinstance(data.xstep,int) and isinstance(data.ystep,int):
        try:
            for i in range(-int((xmax/xstep)),int(xmax/xstep)+1):
                canvas.create_line(i*winWidth//(2*xmax/xstep)+winWidth/2,winHeight/2-mark,
                    i*winWidth//(2*xmax/xstep)+winWidth/2,winHeight/2+mark,fill=data.black)
                canvas.create_text(i*winWidth//(2*xmax/xstep)+winWidth/2,winHeight/2+offset,
                    text="%d" % (i*xstep),fill=data.black)
            for j in range(-int((ymax/ystep)),int(ymax/ystep)+1):
                canvas.create_line(winWidth/2-mark,j*winHeight//(2*ymax/ystep)+winHeight/2,
                    winWidth/2+mark,j*winHeight//(2*ymax/ystep)+winHeight/2,fill=data.black)
                canvas.create_text(winWidth/2-offset,j*winHeight//(2*ymax/ystep)+winHeight/2,
                    text="%d" % (-j*ystep),fill=data.black)
        except:
            pass
    j = 0
    if data.fct[0] != "":
        j = 1
    if data.fct[1] != "":
        j = 2
    if data.fct[2] != "":
        j = 3
    for i in range(j):
        if i == 0: color = data.blue
        elif i == 1: color = data.peach
        elif i == 2: color = data.black
        oldScreenX = oldScreenY = None
        if data.graph == True:
            for screenx in range(winWidth):
                x = (screenx - cx) / scalex
                try: # there's a bug here: screeny referenced before assignment if invalid input for a function
                    y = f(x,data,i) * scaley
                    screeny = cy - y
                    if (oldScreenX != None):
                        canvas.create_line(oldScreenX, oldScreenY, screenx, screeny, fill=color,width=2)
                except: pass
                (oldScreenX, oldScreenY) = (screenx,screeny)

def drawLinearAnimations(canvas,data):
    matrixSize = 60
    # draw axes
    winWidth,winHeight = data.width,data.height
    (cx, cy) = (winWidth/2, winHeight/2)
    canvas.create_line(0, cy, winWidth, cy,width=5,fill=data.black)
    canvas.create_line(cx, 0, cx, winHeight,width=5,fill=data.black)
    # draw all gridlines in gray to be compared to for afterwards
    sep = 45
    for i in range(-20,21):
        if i != 0:
            canvas.create_line(data.width//2+i*sep,0,data.width//2+i*sep,
                data.height,fill="seashell3")
            canvas.create_line(0,data.height//2+i*sep,data.width,
                data.height//2+i*sep,fill="seashell3")
    # draw a vector
    Ox,Oy = data.width//2,data.height//2 # origin
    for i in range(len(data.vectorList)):
        canvas.create_line(Oy,Ox,Ox+data.width//12*data.vectorList[i][0],
        Oy-data.width//12*data.vectorList[i][1],width=9,fill=data.black) # minus for y since up is down
    for i in range(len(data.matrixAlist)):
        for j in range(len(data.matrixAlist[0])):
            canvas.create_text(data.width//12+(i+0.5)*matrixSize,
                data.height//12+(j+0.5)*matrixSize,
                text=str(data.matrixAlist[i][j]),font="Arial 15 bold",fill=data.black)
    for i in range(-20,21):
        canvas.create_line(data.xValuesH[i][0],data.yValuesH[i][0],
                           data.xValuesH[i][1],data.yValuesH[i][1],fill=data.blue)
        canvas.create_line(data.xValuesV[i][0],data.yValuesV[i][0],
                           data.xValuesV[i][1],data.yValuesV[i][1],fill=data.blue)
    canvas.create_rectangle(0,0,data.width//8,data.height//10,fill=data.peach,width=0)
    canvas.create_text(data.width//16,data.height//20,fill=data.white,font="Arial",text="home")
    canvas.create_line(data.width//12,data.height//12,data.width//12,
        data.height//12+2*matrixSize,width=15,fill=data.black)
    canvas.create_line(data.width//12+2*matrixSize,data.height//12,
        data.width//12+2*matrixSize,data.height//12+2*matrixSize,width=15,fill=data.black)

def graphRedrawAll(canvas,data):
    canvas.create_rectangle(0,0,data.width,data.height,fill=data.white,width=0)
    canvas.create_rectangle(0,0,data.width//8,data.height//10,fill=data.peach,width=0)
    canvas.create_text(data.width//16,data.height//20,fill=data.white,font="Arial",text="home")
    if data.graphMode == None: drawGraphMenu(canvas,data)
    canvas.create_text(data.width//2,data.height//12,text="G r a p h i n g",font="Arial 14 bold",fill=data.black)
    if data.graphMode == "real":
        canvas.create_text(data.width//2,data.height-data.margin,text="click to return home; type functions; hit enter to graph; use right arrow to scale",font="Arial",fill=data.black)
    if data.graphMode == "r2":
        canvas.create_text(data.width//2-4,data.height-data.margin,text="type numbers to make a matrix for your linear transformation, 'enter' to graph",font="Arial",fill=data.black)
    if data.graphMode == "real": drawRealFunctions(canvas,data)
    if data.graphMode == "r2": drawLinearAnimations(canvas,data)

######
# stats functions
######

def inBox(x0,y0,x1,y1,x,y):
    if x < x1 and x > x0 and y < y1 and y > y0:
        return True
    return False

def statsMousePressed(event,data):
    if data.statsMode == "menu":
        if inBox(data.margin,data.startHeight,data.width//2-data.margin,
        data.startHeight*4,event.x,event.y):
            data.statsMode = "OVS"
        elif inBox(data.width//2+data.margin,data.startHeight,
        data.width-data.margin,data.startHeight*4,event.x,event.y):
            data.statsMode = "2VS"
    if inBox(0,0,data.width//8,data.height//10,event.x,event.y):
        init(data)

def OVSKeyPressed(event,data):
    if event.keysym == "c": # for calculate
        total = 0
        for entry in data.univariateData:
            total += entry
        data.mean = round(total / len(data.univariateData),2)
        newTotal = 0
        for entry in data.univariateData:
            newTotal += (entry-data.mean)**2
        data.SD = ((newTotal)/(len(data.univariateData)-1))**0.5
        data.SD = round(data.SD,2)
        # chop off those stupid extra digits
        data.univariateData.sort()
        l = len(data.univariateData)
        if l%2 == 1:
            data.median = data.univariateData[l//2]
        else:
            data.median=(data.univariateData[l//2]+data.univariateData[l//2-1])/2
        data.min = data.univariateData[0]
        data.max = data.univariateData[-1]
        qrt1 = data.univariateData[(l-1)//4]
        qrt3 = data.univariateData[3*(l-1)//4]
        data.Q1 = qrt1 # + data.univariateData[(l)//4]*((l)/4%1)
        data.Q3 = qrt3 # + data.univariateData[3*(l)//4]*((3*(l)/4%1))
        for item in data.univariateData:
            if item < data.Q1 - 1.5 * (data.Q3-data.Q1) or item > data.Q3 + 1.5 * (data.Q3-data.Q1):
                data.outliers += [item]
        for i in range(len(data.univariateData)):
            if data.univariateData[i] not in data.outliers:
                data.minFX = data.univariateData[i]
                break
        for i in range(len(data.univariateData)):
            if data.univariateData[-i-1] not in data.outliers:
                data.maxFX = data.univariateData[-i-1]
                break
        return
    if event.keysym == "Return":
        data.tracker = 0
        data.entry = 0
        data.point, data.count  = False, 0
        return
    if data.tracker == 0:
        data.univariateData += [data.entry]
    if data.point == False:
        if event.keysym in "0123456789":
            data.univariateData[-1] *= 10
            data.univariateData[-1] += int(event.keysym)
            data.tracker = 1
    if data.point == True:
        if event.keysym in "0123456789":
            data.count += 1
            data.univariateData[-1] += int(event.keysym) * 10 ** (-data.count)
            data.tracker = 1
    if event.keysym == "period":
        data.point = True

def TVSKeyPressed(event,data):
    if len(data.L2) == len(data.L1) and len(data.L2) > 0 and event.keysym == "c":
        for i in range(len(data.L2)):
            data.L1squared += [(data.L1[i])**2]
            data.L2squared += [(data.L2[i])**2]
            data.L1L2 += [data.L1[i]*data.L2[i]]
        n = len(data.L1)
        num = n*sum(data.L1L2) - sum(data.L1)*sum(data.L2)
        den1 = n*sum(data.L1squared)-(sum(data.L1))**2
        den2 = n*sum(data.L2squared)-(sum(data.L2))**2
        data.r = num/((den1*den2)**0.5)
        L1avg = sum(data.L1) / len(data.L1)
        L2avg = sum(data.L2) / len(data.L2)
        SDL1,SDL2 = 0,0
        for i in range(len(data.L1)):
            SDL1 += (data.L1[i]-L1avg)**2
            SDL2 += (data.L2[i]-L2avg)**2
        SDL1 = (SDL1/n)**0.5
        SDL2 = (SDL2/n)**0.5
        data.b1 = data.r * SDL2 / SDL1
        data.b0 = L2avg - data.b1 * L1avg
    if event.keysym == "period":
        data.point = True
    if event.keysym == "s":
        data.dim = 1
        data.tracker = 0
        data.count = -1 # this is a weird thing to hard code but it works
        return
    if data.dim == 0:
        if data.tracker == 0:
            data.entry = 0
            data.L1 += [data.entry]
        if data.point == False:
            if event.keysym in "0123456789":
                data.L1[-1] *= 10
                data.L1[-1] += int(event.keysym)
                data.tracker = 1
        if data.point == True:
            if event.keysym in "0123456789":
                data.count += 1
                data.L1[-1] += int(event.keysym) * 10 ** (-data.count)
                data.tracker = 1
    if data.dim == 1:
        if data.tracker == 0:
            data.entry = 0
            data.L2 += [data.entry]
        if data.point == False:
            if event.keysym in "0123456789":
                data.L2[-1] *= 10
                data.L2[-1] += int(event.keysym)
                data.tracker = 1
        if data.point == True:
            if event.keysym in "0123456789":
                data.count += 1
                data.L2[-1] += int(event.keysym) * 10 ** (-data.count)
                data.tracker = 1
    if event.keysym == "Return": 
        data.tracker,data.count = 0,0
        data.point = False

def statsKeyPressed(event,data):
    if data.statsMode == "OVS":
        OVSKeyPressed(event,data)
    if data.statsMode == "2VS":
        TVSKeyPressed(event,data)

def statsTimer(data):
    pass

def drawStatsMenu(canvas,data):
    canvas.create_rectangle(data.margin,data.startHeight,data.width//2-data.margin,
    data.startHeight*4,fill=data.blue,width=0)
    canvas.create_rectangle(data.width//2+data.margin,data.startHeight,
    data.width-data.margin,data.startHeight*4,fill=data.yellow,width=0)
    canvas.create_text(data.width//4,5*data.startHeight//2,
    text="one-variable stats",font="Arial",fill=data.white)
    canvas.create_text(3*data.width//4,5*data.startHeight//2,
    text="two-variable stats",font="Arial",fill=data.black)

def drawOVSscreen(canvas,data):
    canvas.create_text(data.width//4,data.height//4,text="univariate data",font="Arial",fill=data.black)
    canvas.create_line(data.width//4-5*data.margin,data.height//4+data.margin,
    data.width//4+5*data.margin,data.height//4+data.margin,fill=data.black)
    canvas.create_text(3*data.width//4,data.height//4,text="one variable stats",
        font="Arial",fill=data.black)
    canvas.create_line(3*data.width//4-5*data.margin,data.height//4+data.margin,
    3*data.width//4+5*data.margin,data.height//4+data.margin,fill=data.black)
    for i in range(len(data.univariateData)):
        canvas.create_text(data.width//4,data.height//4+(i+2)*data.margin*1.5,
        text=str(data.univariateData[i]),font="Arial",fill=data.black)
    canvas.create_text(3*data.width//4,data.height//4+3*data.margin,
    text="mean: "+str(data.mean),font="Arial",fill=data.black)
    canvas.create_text(3*data.width//4,data.height//4+4.5*data.margin,
    text="standard deviation: "+str(data.SD),font="Arial",fill=data.black)
    canvas.create_text(3*data.width//4,data.height//4+6*data.margin,
    text="min: "+str(data.min),font="Arial",fill=data.black)
    canvas.create_text(3*data.width//4,data.height//4+7.5*data.margin,
    text="Q1: "+str(data.Q1),font="Arial",fill=data.black)
    canvas.create_text(3*data.width//4,data.height//4+9*data.margin,
    text="median: "+str(data.median),font="Arial",fill=data.black)
    canvas.create_text(3*data.width//4,data.height//4+10.5*data.margin,
    text="Q3: "+str(data.Q3),font="Arial",fill=data.black)
    canvas.create_text(3*data.width//4,data.height//4+12*data.margin,
    text="max: "+str(data.max),font="Arial",fill=data.black)
    canvas.create_text(3*data.width//4,data.height//4+13.5*data.margin,
    text="IQR: [%d,%d]" % (data.Q1,data.Q3),font="Arial",fill=data.black)
    canvas.create_text(data.width//2,data.height*5//7-50,
    text="boxplot:",fill=data.black,font="Arial 20 bold")
    nM = 250 # new Margin
    h = 50 # boxplot height
    if data.Q3 != 0:
        canvas.create_line(data.width//5,3*data.height//4-h,data.width//5,
            3*data.height//4+h,fill=data.black)
        canvas.create_line(4*data.width//5,3*data.height//4-h,4*data.width//5,
            3*data.height//4+h,fill=data.black)
        canvas.create_line(data.width//5,3*data.height//4,4*data.width//5,
            3*data.height//4,fill=data.black)
        y0,y1 = 3*data.height//4-h, 3*data.height//4+h
        x0 = (data.Q1-data.minFX)/(data.maxFX-data.minFX) * 3*data.width//5 + data.width//5
        x1 = (data.Q3-data.minFX)/(data.maxFX-data.minFX) * 3*data.width//5 + data.width//5
        canvas.create_rectangle(x0,y0,x1,y1,fill=data.yellow,width=0)
        canvas.create_line((data.median-data.minFX)/(data.maxFX-data.minFX) * 3*data.width//5 + data.width//5,
            y0,(data.median-data.minFX)/(data.maxFX-data.minFX) * 3*data.width//5 + data.width//5,y1,fill=data.black)
        newHeight = 3*data.height//4+h+20
        canvas.create_text(data.width//10,newHeight,text="min",font="Arial",fill=data.black)
        canvas.create_text(9*data.width//10,newHeight,text="max",font="Arial",fill=data.black)
        canvas.create_text(x0,newHeight,text="Q1",font="Arial",fill=data.black)
        canvas.create_text(x1,newHeight,text="Q3",font="Arial",fill=data.black)
        location = (data.median-data.minFX)/(data.maxFX-data.minFX) * 3*data.width//5 + data.width//5
        canvas.create_text((0.3*(x0+x1)/2+0.7*location),newHeight,text="median",font="Arial",fill=data.black)
    scale = data.width//5
    for outlier in data.outliers:
        if outlier < data.Q1:
            canvas.create_text(data.width//5+scale*(outlier-data.minFX)/data.max,
                3*data.height//4+8,text="*",font="Arial 40 bold",fill=data.peach)
        elif outlier > data.Q3:
            canvas.create_text(4*data.width//5+scale*(outlier-data.maxFX)/data.max,
                3*data.height//4+8,text="*",font="Arial 40 bold",fill=data.peach)

def draw2VSScreen(canvas,data):
    canvas.create_text(data.width//4,data.height//4,text="bivariate data",font="Arial",fill=data.black)
    canvas.create_line(data.width//4-5*data.margin,data.height//4+data.margin,
    data.width//4+5*data.margin,data.height//4+data.margin,fill=data.black)
    canvas.create_line(data.width//4,data.height//4+data.margin,
        data.width//4,data.height//2,fill=data.black)
    canvas.create_text(3*data.width//4,data.height//4,text="linear regression",font="Arial",fill=data.black)
    canvas.create_line(3*data.width//4-5*data.margin,data.height//4+data.margin,
    3*data.width//4+5*data.margin,data.height//4+data.margin,fill=data.black)
    nM = 20 # new margin
    for i in range(len(data.L1)):
        canvas.create_text(data.width//4-nM,data.height//4+2*data.margin*(i+1),
            text=str(data.L1[i]),font="Arial",fill=data.black)
    for j in range(len(data.L2)):
        canvas.create_text(data.width//4+nM,data.height//4+2*data.margin*(j+1),
            text=str(data.L2[j]),font="Arial",fill=data.black)
    if len(data.L2) == len(data.L1) and len(data.L2) > 0:
        canvas.create_line(data.width//3,5*data.height//6,2*data.width//3,
            5*data.height//6,fill=data.black)
        canvas.create_line(data.width//3,5*data.height//6,data.width//3,
            data.height//2,fill=data.black)
        unitx = data.width//3 / max(data.L1)
        unity = data.width//3 / max(data.L2)
        startx = data.width//3
        starty = 5*data.width//6
        for i in range(len(data.L1)):
            rad = 2
            canvas.create_oval(startx+unitx*data.L1[i]-rad,starty-unity*data.L2[i]-rad,
                startx+unitx*data.L1[i]+rad,starty-unity*data.L2[i]+rad,width=0,
                fill=data.peach)
        size = 2*data.margin
        canvas.create_text(3*data.width//4,data.height//4+size,
            text="b0 = "+str(data.b0),font="Arial",fill=data.black)
        canvas.create_text(3*data.width//4,data.height//4+2*size,
            text="b1 = "+str(data.b1),font="Arial",fill=data.black)
        canvas.create_text(3*data.width//4,data.height//4+3*size,
            text="r = "+str(data.r),font="Arial",fill=data.black)
        canvas.create_text(3*data.width//4,data.height//4+4*size,
            text="y = b0 + b1 * x",font="Arial",fill=data.black)
        # make sure regression line doesn't leave axes !
        slope = data.b1 * unity / unitx
        x0,y0,x1,y1 = data.width//3,starty-unity*data.b0,2*data.width//3,starty-unity*data.b0-data.width//3/unitx*data.b1*unity
        if y0 < data.height//2:
            y0 = data.height//2
            x0 = x1 + (y1-y0) / slope
        if y0 > 5*data.height//6:
            y0 = 5*data.height//6
            x0 = x1 + (y1-y0) / slope
        if data.b1 != 0:
            canvas.create_line(x0,y0,x1,y1,width=2,fill=data.blue) # regression line
        mark = 2
        for i in range(5):
            xMark = data.width//3+(i+1)*data.width//15
            yMark = data.height//2+i*data.height//15
            canvas.create_line(xMark,5*data.width//6-mark,xMark,5*data.width//6+mark,fill=data.black) # hash-marks on x axis
            canvas.create_line(data.width//3-mark,yMark,data.width//3+mark,yMark,fill=data.black) # hash-marks on y axis

def statsRedrawAll(canvas,data):
    canvas.create_rectangle(0,0,data.width,data.height,fill=data.white,width=0)
    canvas.create_rectangle(0,0,data.width//8,data.height//10,fill=data.peach,width=0)
    canvas.create_text(data.width//16,data.height//20,fill=data.white,font="Arial",text="home")
    canvas.create_text(data.width//2,data.height//12,text="S t a t i s t i c s",font="Arial 20 bold",fill=data.black)
    if data.statsMode != "menu":
        if data.statsMode == "OVS":
            canvas.create_text(data.width//2,data.height-data.margin,text="type numbers to create entries; hit 'enter' to move to the next entry, 'c' to compute",font="Arial",fill=data.black)
        elif data.statsMode == "2VS":
            canvas.create_text(data.width//2,data.height-3*data.margin,text="type numbers for entries; hit 'enter' to move to the next entry",font="Arial",fill=data.black)
            canvas.create_text(data.width//2,data.height-data.margin,text="'s' to switch cols, 'c' to compute",font="Arial",fill=data.black)
    if data.statsMode == "menu":
        drawStatsMenu(canvas,data)
    if data.statsMode == "OVS":
        drawOVSscreen(canvas,data)
    if data.statsMode == "2VS":
        draw2VSScreen(canvas,data)

#####
# help functions (not helper functions)
#####

def helpMousePressed(event,data):
    data.mode = "menu"

def helpKeyPressed(event,data):
    print(data.helpyValuesH[0])

def helpTimer(data):
    pass

def helpRedrawAll(canvas,data):
    width,height = data.width,data.height
    for i in range(-40,41):
        canvas.create_line(data.helpxValuesH[i][0],data.helpyValuesH[i],
                           data.helpxValuesH[i][1],fill=data.peach)
        canvas.create_line(data.helpxValuesH[i][1],data.helpyValuesH[i],
                           data.helpxValuesH[i][0],fill=data.yellow)
    canvas.create_text(data.width//2+10,data.height//2-23,text='''
    This application was made possible by 
    the wonderful course staff of 15-112.
    Without their motivation, my project could 
    never have made it this far. Special thanks
    to Christine Zhang, my mentor, for 
    inspiration on design, and Miranda Chen, who 
    helped me write the Gaussian Elimination function.
    ''',font="Helvetica 20",fill=data.blue)


####################################
# use the run function as-is
####################################

def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()    

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 100 # milliseconds
    init(data)
    # create the root and the canvas
    root = Tk()
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

run(550,550)