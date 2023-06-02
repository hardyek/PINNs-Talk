from manim import *
from manim_slides import Slide
import math

class Presentation(Slide):
    def construct(self):
        ProblemStatementTitle = Tex("The Heat Equation").to_edge(UP + LEFT)

        HeatEQN_1 = MathTex(r"\frac{\partial u}{\partial t}",
                            r"=",
                            r"\frac{\partial^{2}u}{\partial x^{2}}",
                            r"+"
                            r"\frac{\partial^{2}u}{\partial y^{2}}"
        ).next_to(ProblemStatementTitle, DOWN).to_edge(LEFT)

        Solution = MathTex(r"u(x,y,t) = \text{Temperature}_{x,y,t}").next_to(HeatEQN_1, DOWN).scale(0.8).shift(RIGHT * 0.7).to_edge(LEFT)

        InitialCondition = MathTex(r"u(x,y,0) = \frac{1}{2\pi \sigma^{2}}e^{-\frac{(x-x_{0})^{2} + (y-y_{0})^{2}}{2\sigma^{2}}}").next_to(Solution, DOWN).shift(RIGHT * 0.55).scale(0.8).to_edge(LEFT)
        InitialConditionImage = ImageMobject("initial_conditions.png").scale(1.5).to_edge(RIGHT)

        BoundaryConditions0 = MathTex(r"u(0,y,t) = u(x,0,t) = 0").next_to(InitialCondition, DOWN).scale(0.8).to_edge(LEFT)
        BoundaryConditions1 = MathTex(r"u(1,y,t) = u(x,1,t) = 0").next_to(BoundaryConditions0, DOWN).scale(0.8).to_edge(LEFT)

        Approx = MathTex(r"u(x,y,t) \approx f(x,y,t,\text{learned parameters})").to_edge(DOWN)

        self.play(FadeIn(ProblemStatementTitle), FadeIn(HeatEQN_1))
        self.next_slide()

        self.play(FadeIn(Solution))
        self.next_slide()

        self.play(FadeIn(InitialCondition), FadeIn(InitialConditionImage))
        self.next_slide()

        self.play(FadeIn(BoundaryConditions0), FadeIn(BoundaryConditions1))
        self.next_slide()

        self.play(FadeIn(Approx))



        self.next_slide()
        self.clear()



        NeuralNetworkTitle = Tex("Function Approximator").to_edge(UP + LEFT)

        SolverFunction = MathTex(r"f(x,y,t,\text{learnable parameters})").next_to(NeuralNetworkTitle, RIGHT)

        WeightsMat = MathTex(r"\begin{bmatrix} w_{1,1} & \cdots & w_{1,n} \\ \vdots & \ddots & \vdots \\ w_{m,1} & \cdots & w_{m,n} \end{bmatrix}").scale(0.8).to_edge(LEFT)
        InputsVec = MathTex(r"\begin{bmatrix} x \\ y \\ t \end{bmatrix}").scale(0.8).next_to(WeightsMat, RIGHT)
        BiasVec = MathTex(r" + \begin{bmatrix} b_{1} \\ \vdots \\ b_{m} \end{bmatrix}").scale(0.8).next_to(InputsVec, RIGHT)
        OutVec = MathTex(r"= \begin{bmatrix} a_{1} \\ \vdots \\ a_{m} \end{bmatrix}").scale(0.8).next_to(BiasVec, RIGHT)

        TanhAxes = Axes(x_range=[-5,5,1], y_range=[-1.25,1.25,0.25], axis_config={"include_numbers":True},x_length=10,y_length=10).scale(0.5).to_edge(RIGHT + DOWN)
        Tanh = TanhAxes.plot(lambda x: math.tanh(x), x_range=[-5,5], use_smoothing=False, color=TEAL_B)
        TanhText = MathTex(r"y = \tanh(x)").scale(0.7).next_to(TanhAxes, UP)

        OutVec2 = MathTex(r"\tanh\begin{bmatrix} a_{1} \\ \vdots \\ a_{m} \end{bmatrix}").scale(0.8).to_edge(LEFT)
        OutVecTanh = MathTex(r"= \begin{bmatrix} z_{1} \\ \vdots \\ z_{m} \end{bmatrix}").scale(0.8).next_to(OutVec2,RIGHT)

        self.play(FadeIn(NeuralNetworkTitle), FadeIn(SolverFunction))
        self.next_slide()
        self.play(FadeIn(WeightsMat), FadeIn(InputsVec))
        self.next_slide()
        self.play(FadeIn(BiasVec), FadeIn(OutVec))
        self.next_slide()
        self.play(FadeIn(TanhAxes),Write(Tanh),FadeIn(TanhText))
        self.next_slide()
        self.play(Transform(WeightsMat, OutVec2), Transform(InputsVec, OutVecTanh), FadeOut(BiasVec), FadeOut(OutVec))



        self.next_slide()
        self.clear()



        GradientDescentTitle = Tex("Gradient Descent").to_edge(UP + LEFT)

        GradientDescentFormal = MathTex(r"a_{n+1} = a_{n} - \alpha\frac{df(a)}{da}").next_to(GradientDescentTitle, DOWN)

        GDAxes = Axes(x_range=[-5,5,1], y_range=[0,26,5], axis_config={"include_numbers":True},x_length=10,y_length=10).scale(0.5).to_edge(RIGHT)
        ax2Graph = GDAxes.plot(lambda x: x ** 2, x_range=[-5,5], use_smoothing=False, color=TEAL_B)

        ax = MathTex(r"f(a) = a^{2}").next_to(GDAxes, UP).scale(0.7).shift(LEFT * 0.75)
        aprimex = MathTex(r"\frac{df(a)}{da} = 2a").scale(0.7).next_to(ax, RIGHT)

        GraphGroup = VGroup(GDAxes,ax2Graph,ax,aprimex)
        GraphGroup.shift(LEFT)

        Alpha = MathTex(r"\alpha = 0.1").scale(0.7).next_to(GradientDescentFormal, DOWN)
        A0 = MathTex(r"a_{0} = 4.5").scale(0.7).next_to(Alpha, DOWN)
        A1L1 = MathTex(r"a_{1} =",r"a_{0}",r"-0.1",r"(2a_{0})").scale(0.7).next_to(A0, DOWN)
        A1L2 = MathTex(r"a_{1} = 4.5 - 0.9 = 3.6").scale(0.7).next_to(A1L1, DOWN)
        A2 = MathTex(r"a_{2} = 3.6 - 0.72 = 2.88").scale(0.7).next_to(A1L2, DOWN)
        A3 = MathTex(r"a_{3} = 2.304").scale(0.7).next_to(A2, DOWN)
        A4 = MathTex(r"a_{4} = 1.843").scale(0.7).next_to(A3, DOWN)
        A5 = MathTex(r"a_{5} = 1.475").scale(0.7).next_to(A4, DOWN)
        Vdots = MathTex(r"\ddots").scale(0.7).next_to(A5, DOWN)
        A20 = MathTex(r"a_{20} = 0.052").scale(0.7).next_to(Vdots, DOWN)

        dot = Dot(color=RED, radius=0.1)
        dot.move_to(GDAxes.coords_to_point(4.5, 4.5 ** 2))

        tangent0 = GDAxes.plot(lambda x: 2 * (4.5) * x - (4.5 ** 2), color=YELLOW_B)
        tangent1 = GDAxes.plot(lambda x: 2 * (3.6) * x - (3.6 ** 2), color=YELLOW_B)
        tangent2 = GDAxes.plot(lambda x: 2 * (2.88) * x - (2.88 ** 2), color=YELLOW_B)
        tangent3 = GDAxes.plot(lambda x: 2 * (2.304) * x - (2.304 ** 2), color=YELLOW_B)
        tangent4 = GDAxes.plot(lambda x: 2 * (1.843) * x - (1.843 ** 2), color=YELLOW_B)
        tangent5 = GDAxes.plot(lambda x: 2 * (1.475) * x - (1.475 ** 2), color=YELLOW_B)

        self.play(FadeIn(GradientDescentTitle), FadeIn(GradientDescentFormal))
        self.next_slide()
        self.play(FadeIn(GDAxes), FadeIn(ax), FadeIn(aprimex), Write(ax2Graph))
        self.next_slide()
        self.play(FadeIn(Alpha), FadeIn(A0), FadeIn(dot))
        self.next_slide()
        self.play(FadeIn(tangent0), FadeIn(A1L1))
        self.next_slide()
        self.play(dot.animate.move_to(GDAxes.coords_to_point(3.6, 3.6 ** 2)), FadeIn(A1L2))
        self.next_slide()

        self.start_loop()
        self.play(FadeIn(A2),FadeOut(tangent0),FadeIn(tangent1))
        self.play(dot.animate.move_to(GDAxes.coords_to_point(2.88, 2.88 ** 2)))
        self.play(FadeIn(A3),FadeOut(tangent1),FadeIn(tangent2))
        self.play(dot.animate.move_to(GDAxes.coords_to_point(2.304, 2.304 ** 2)))
        self.play(FadeIn(A4),FadeOut(tangent2),FadeIn(tangent3))
        self.play(dot.animate.move_to(GDAxes.coords_to_point(1.843, 1.843 ** 2)))
        self.play(FadeIn(A5),FadeOut(tangent3),FadeIn(tangent4))
        self.play(dot.animate.move_to(GDAxes.coords_to_point(1.475, 1.475 ** 2)))
        self.play(FadeIn(Vdots),FadeOut(tangent4),FadeIn(tangent5))
        self.play(FadeIn(A20))
        self.play(dot.animate.move_to(GDAxes.coords_to_point(0.052, 0.052 ** 2)), FadeOut(tangent5))
        self.play(FadeOut(A2,A3,A4,A5,Vdots,A20), FadeIn(tangent0), dot.animate.move_to(GDAxes.coords_to_point(3.6, 3.6 ** 2)))
        self.end_loop()



        self.clear()



        ChainRuleTitle = Tex("Chain Rule").to_edge(UP + LEFT)
        ChainRuleFormal = MathTex(r"\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}")

        Car = SVGMobject(file_name="car.svg", fill_color=RED_C).to_edge(UP + LEFT).scale(0.25).shift(DOWN + LEFT * 1.7)
        Bike = SVGMobject(file_name="bike.svg", fill_color=GREEN_C).next_to(Car, DOWN * 1.5).scale(0.25)
        Man = SVGMobject(file_name="man.svg", fill_color=BLUE_C).next_to(Bike, DOWN * 1.5).scale(0.25)

        CarLine = NumberLine(x_range=[0,10,1], color=RED_A, include_numbers=True).next_to(Car, RIGHT)
        BikeLine = NumberLine(x_range=[0,10,1], color=GREEN_A, include_numbers=True).next_to(Bike, RIGHT)
        ManLine = NumberLine(x_range=[0,10,1], color=BLUE_A, include_numbers=True).next_to(Man, RIGHT)

        x_parameter = ValueTracker(0)
        CarDot = Dot(color=WHITE).add_updater(lambda mob: mob.move_to(CarLine.number_to_point(x_parameter.get_value() * 4)))
        BikeDot = Dot(color=WHITE).add_updater(lambda mob: mob.move_to(BikeLine.number_to_point(x_parameter.get_value() * 2)))
        ManDot = Dot(color=WHITE).add_updater(lambda mob: mob.move_to(ManLine.number_to_point(x_parameter.get_value())))

        DC_DB_1 = MathTex(r"\frac{dC}{dB} = 2", color=RED_C).next_to(CarLine, RIGHT)
        DB_DM_1 = MathTex(r"\frac{dB}{dM} = 2", color=GREEN_C).next_to(BikeLine, RIGHT)

        DC_DM = MathTex(r"\frac{dC}{dM}",color=BLUE_C).next_to(ManLine, DOWN).shift(LEFT * 0.5)
        equals = MathTex(r"=").next_to(DC_DM, RIGHT)
        DC_DB = MathTex(r"\frac{dC}{dB}", color=RED_C).next_to(equals, RIGHT)
        cdot = MathTex(r"\cdot").next_to(DC_DB, RIGHT)
        DB_DM = MathTex(r"\frac{dB}{dM}", color=GREEN_C).next_to(cdot, RIGHT)
        result = MathTex(r"= 4").next_to(DB_DM, RIGHT)

        self.play(FadeIn(ChainRuleTitle), FadeIn(ChainRuleFormal))
        self.next_slide()
        self.play(ChainRuleFormal.animate.to_edge(UP + RIGHT), FadeIn(ManLine,BikeLine,CarLine,Car,Bike,Man,CarDot,BikeDot,ManDot))
        self.next_slide()
        self.start_loop()
        self.play(x_parameter.animate.set_value(2.5), runtime=2.5, rate_func=rate_functions.linear)
        self.wait(0.5)
        self.end_loop()
        self.play(FadeIn(DC_DB_1, DB_DM_1))
        self.next_slide()
        self.play(FadeIn(DC_DM,equals,DC_DB,cdot,DB_DM,result))
        self.next_slide()



        self.clear()



        AutoDiffTitle = Tex("Chain rule for Automatic Differentiation").to_edge(UP + LEFT)
        LEquation = MathTex(r"L = \sin(wx + b)").scale(0.8).next_to(AutoDiffTitle, DOWN).to_edge(LEFT)
        LBroken = MathTex(r"L = \sin(a), a = wx + b").scale(0.8).next_to(LEquation, DOWN).to_edge(LEFT)
        DLDa = MathTex(r"\frac{dL}{da} = \cos(a)").scale(0.8).next_to(LBroken, DOWN).to_edge(LEFT)
        DaDw = MathTex(r"\frac{\partial a}{\partial w} = x").scale(0.8).next_to(DLDa, DOWN).to_edge(LEFT)
        DaDx = MathTex(r"\frac{\partial a}{\partial x} = w").scale(0.8).next_to(DaDw, DOWN).to_edge(LEFT)
        DaDb = MathTex(r"\frac{\partial a}{\partial b} = 1").scale(0.8).next_to(DaDx, DOWN).to_edge(LEFT)
        
        GraphB = MathTex(r"b").to_edge(DOWN + LEFT)
        BBox = SurroundingRectangle(GraphB)

        GraphX = MathTex(r"x").next_to(GraphB, UP)
        XBox = SurroundingRectangle(GraphX)

        GraphW = MathTex(r"w").next_to(GraphX, UP)
        WBox = SurroundingRectangle(GraphW)

        Times = MathTex(r"\times").next_to(GraphX, RIGHT).shift(RIGHT).shift(UP*0.2)
        TimesCirc = Circle().surround(Times)

        Addition = MathTex(r"+").next_to(Times, RIGHT).shift(RIGHT)
        AdditionCirc = Circle().surround(Addition)

        Sin = MathTex(r"\sin").next_to(Addition, RIGHT).shift(RIGHT)
        SinCirc = Circle().surround(Sin)

        GraphL = MathTex(r"L").next_to(Sin, RIGHT).shift(RIGHT)
        LBox = SurroundingRectangle(GraphL)

        WTimesLine = Line(WBox.get_right(),TimesCirc.get_left())
        XTimesLine = Line(XBox.get_right(),TimesCirc.get_left())
        BAdditionLine = Line(BBox.get_right(), AdditionCirc.get_left())
        TimesAdditionLine = Line(TimesCirc.get_right(), AdditionCirc.get_left())
        AdditionSinLine = Line(AdditionCirc.get_right(), SinCirc.get_left())
        SinLLine = Line(SinCirc.get_right(), LBox.get_left())

        GraphA = MathTex(r"a").next_to(AdditionSinLine, UP)

        GraphGroup = VGroup(GraphB, GraphX, GraphW, Times,TimesCirc, Addition, AdditionCirc, Sin, SinCirc, GraphL, WTimesLine, XTimesLine, BAdditionLine, TimesAdditionLine, AdditionSinLine, SinLLine, GraphA)
        GraphGroup.shift(RIGHT * 2.8)


        BackPropCode = Code(code = """def backward(self):
        graph = []
        visited = set()
        def build_graph(s):
            if s not in visited:
                visited.add(s)
                for node in s.dependencies:
                    build_graph(node)
                graph.append(s)
        build_graph(self)

        for s in graph:
            s.zero_grad()

        self.grad = 1
        for s in reversed(graph):
            s.back())""", language="python").scale(0.75).to_edge(RIGHT).shift(LEFT).shift(UP * 0.25)    
        
        OperationCode = Code(code = """def __mul__(self, factor):
        factor = factor if isinstance(factor, Scalar) else Scalar(factor)
        result = Scalar(self.data * factor.data, {self, factor})

        def back():
            self.grad += factor.data * result.grad
            factor.grad += self.data * result.grad

        result.back = back

        return result""", language="python").scale(0.7).to_edge(RIGHT)
        
        self.play(FadeIn(AutoDiffTitle, LEquation))
        self.next_slide()
        self.play(FadeIn(LBroken))
        self.next_slide()
        self.play(FadeIn(DLDa))
        self.next_slide()
        self.play(FadeIn(DaDw, DaDx, DaDb))
        self.next_slide()
        self.play(FadeIn(GraphGroup))
        self.next_slide()
        self.play(FadeIn(BackPropCode))
        self.next_slide()
        self.play(FadeOut(BackPropCode),FadeIn(OperationCode))



        self.next_slide()
        self.clear()



        
        SolutionTitle = Tex("Using these to solve the heat equation.").to_edge(UP + LEFT)
        Solution = MathTex(r"u(x,y,t) \approx f(x,y,t)").to_edge(UP).shift(DOWN)
        Solution2 = MathTex(r"f(x,y,t) \text{ is a model with 2081 trainable parameters.}").scale(0.7).next_to(Solution, DOWN)
        
        MinimiseFunction = MathTex(r"minimise:", r"\sum_{y=1}^{99} \sum_{x=1}^{99} \sum_{t=0}^{100} \left( \frac{\partial f(x,y,t)}{\partial t} - \frac{\partial^{2}f(x,y,t)}{\partial x^{2}} - \frac{\partial^{2}f(x,y,t)}{\partial y^{2}} \right)^{2}" ,r"+ \sum_{x=1}^{99} \sum_{y=1}^{99} \left( f(x,y,0) - \frac{1}{2\pi \sigma^{2}}e^{-\frac{(x-x_{0})^{2} + (y-y_{0})^{2}}{2\sigma^{2}}} \right)^{2}" ,r"+ \sum_{y=0}^{100} \sum_{t=0}^{100} \left( f(0,y,t) \right)^{2} + \sum_{y=0}^{100} \sum_{t=0}^{100} \left( f(100,y,t) \right)^{2} + \sum_{x=0}^{100} \sum_{t=0}^{100} \left( f(x,0,t) \right)^{2} + \sum_{x=0}^{100} \sum_{t=0}^{100} \left( f(x,100,t) \right)^{2}").scale(0.3).to_edge(DOWN).shift(UP * 3)


        TrainingLosses = ImageMobject("Training Losses.png").scale(0.65).to_edge(DOWN)

        EquationBox = SurroundingRectangle(MinimiseFunction[1])
        EquationLine = Line(EquationBox.get_left(), EquationBox.get_right(), color=RED).shift(DOWN * 0.3)

        InitialBox = SurroundingRectangle(MinimiseFunction[2])
        InitialLine = Line(InitialBox.get_left(), InitialBox.get_right(), color=BLUE).shift(DOWN * 0.3)

        BoundaryBox = SurroundingRectangle(MinimiseFunction[3])
        BoundaryLine = Line(BoundaryBox.get_left(), BoundaryBox.get_right(), color=GREEN).shift(DOWN * 0.3)

        EquationLoss = Tex("Equation (PDE) Loss").scale(0.6).next_to(EquationLine, DOWN)
        InitialLoss = Tex("Initial Loss").scale(0.6).next_to(InitialLine, DOWN)
        BoundaryLoss = Tex("Boundary Loss").scale(0.6).next_to(BoundaryLine, DOWN)

        EquationGroup = VGroup(MinimiseFunction, EquationLine, EquationLoss, InitialLine, InitialLoss, BoundaryLine, BoundaryLoss)

        PINN = Tex("$f(x,y,t)$ is refered to as a PINN (Physics Informed Neural Network).").scale(0.6).next_to(EquationGroup, DOWN)

        self.play(FadeIn(SolutionTitle, Solution, Solution2))
        self.next_slide()
        self.play(FadeIn(MinimiseFunction, EquationLine, EquationLoss, InitialLine, InitialLoss, BoundaryLine, BoundaryLoss))
        self.next_slide()
        self.play(FadeIn(PINN))
        self.next_slide()
        self.play(EquationGroup.animate.to_edge(UP).shift(DOWN), FadeOut(PINN, Solution, Solution2))
        self.play(FadeIn(TrainingLosses))
        self.next_slide()




        self.clear()



        ResultsTitle = Tex("Evolution of $f(x,y,t)$ over training.").to_edge(UP + LEFT)

        self.play(FadeIn(ResultsTitle))

