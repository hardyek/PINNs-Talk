from manim_slides import Slide
from manim import *
import math

class Presentation(Slide):
    def construct(self):

        #SLIDE 1 - TITLE
        SLIDE1_Title = Tex("Chain Rule and Gradient Descent")

        self.play(FadeIn(SLIDE1_Title))
        #SLIDE 2 - WHAT IS THE CHAIN RULE
        self.next_slide()
        self.clear()
        #region Slide2
        SLIDE2_Title = Tex("What is the chain rule?").to_corner(UP + LEFT)

        self.play(FadeIn(SLIDE2_Title))
        self.next_slide()
        ChainRuleFormal = MathTex(r"\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}").scale(2.0)

        self.play(FadeIn(ChainRuleFormal))

        self.next_slide()
        self.play(ChainRuleFormal.animate.scale(0.35), runtime=0.5)
        self.play(ChainRuleFormal.animate.to_edge(UP + RIGHT), runtime=0.5)

        car = SVGMobject(file_name="car.svg", fill_color=RED_C).to_edge(UP + LEFT).scale(0.25).shift(DOWN + LEFT * 1.7)
        bike = SVGMobject(file_name="bike.svg", fill_color=GREEN_C).next_to(car, DOWN * 1.5).scale(0.25)
        man = SVGMobject(file_name="man.svg", fill_color=BLUE_C).next_to(bike, DOWN * 1.5).scale(0.25)

        self.add(car, bike, man)
        
        car_line = NumberLine(x_range=[0,10,1], color=RED_A, include_numbers=True).next_to(car, RIGHT)
        bike_line = NumberLine(x_range=[0,10,1], color=GREEN_A, include_numbers=True).next_to(bike, RIGHT)
        man_line = NumberLine(x_range=[0,10,1], color=BLUE_A, include_numbers=True).next_to(man, RIGHT)

        self.play(Write(car_line), Write(bike_line), Write(man_line), runtime=0.5)

        self.next_slide()

        x_parameter = ValueTracker(0)
        car_dot = Dot(color=WHITE).add_updater(lambda mob: mob.move_to(car_line.number_to_point(x_parameter.get_value() * 4)))
        bike_dot = Dot(color=WHITE).add_updater(lambda mob: mob.move_to(bike_line.number_to_point(x_parameter.get_value() * 2)))
        man_dot = Dot(color=WHITE).add_updater(lambda mob: mob.move_to(man_line.number_to_point(x_parameter.get_value())))

        self.add(car_dot, bike_dot, man_dot)

        self.start_loop()
        self.play(x_parameter.animate.set_value(2.5),runtime=3,rate_func=rate_functions.linear)
        self.wait(1)
        self.end_loop()

        DC_DB = MathTex(r"\frac{dC}{dB} = 2", color=RED_C).next_to(car_line, RIGHT).shift(DOWN * .6)

        self.play(ChainRuleFormal.animate.scale(1), runtime=0.1)
        self.add(DC_DB)

        self.next_slide()

        DB_DM = MathTex(r"\frac{dB}{dM} = 2", color=GREEN_C).next_to(bike_line, RIGHT).shift(DOWN * .6)

        self.play(ChainRuleFormal.animate.scale(1), runtime=0.1)
        self.add(DB_DM)

        self.next_slide()

        DC_DM = MathTex(r"\frac{dC}{dM}",color=BLUE_C).next_to(man_line, DOWN).shift(LEFT * 0.5)
        equals = MathTex(r"=").next_to(DC_DM, RIGHT)
        DC_DB = MathTex(r"\frac{dC}{dB}", color=RED_C).next_to(equals, RIGHT)
        cdot = MathTex(r"\cdot").next_to(DC_DB, RIGHT)
        DB_DM = MathTex(r"\frac{dB}{dM}", color=GREEN_C).next_to(cdot, RIGHT)
        result = MathTex(r"= 4").next_to(DB_DM, RIGHT)

        self.play(ChainRuleFormal.animate.scale(1), runtime=0.1)
        self.add(DC_DM,equals,DC_DB, cdot, DB_DM)

        self.next_slide()

        self.play(ChainRuleFormal.animate.scale(1), runtime=0.1)
        self.add(result)
        #endregion
        #SLIDE 3 - FUNCTIONAL CONTEXT
        self.next_slide()
        self.clear()
        #region Slide3
        SLIDE2_Title = Tex("How can we use it?").to_corner(UP + LEFT)

        self.play(FadeIn(SLIDE2_Title))

        Equation = MathTex(r"y =", r"\sin(", r"\cos(", r"x^{2}))").to_edge(UP).shift(DOWN * 1.5)

        Line1 = MathTex(r"y =", r"\sin(", r"\cos(", r"x^{2}))").next_to(Equation, DOWN)

        a_def = MathTex(r"a = x^{2}").next_to(Line1, DOWN * 2.2)

        Line2 = MathTex(r"y =", r"\sin(", r"\cos(",r"a))").next_to(Equation, DOWN)

        b_def = MathTex(r"b = \cos(a)").next_to(a_def, DOWN * 2.2)

        Line3 = MathTex(r"y" r"=", r"\sin(", r"b)").next_to(Equation, DOWN*1.15)

        c_def = MathTex(r"c = \sin(b)").next_to(b_def, DOWN * 2.2)

        Line4 = MathTex(r"y", r"=", r"c", r"").next_to(Equation, DOWN*1.3)

        self.play(Write(Equation),runtime=0.25)
        self.next_slide()

        self.add(Line1)
        self.wait(1)
        self.add(a_def)
        self.play(TransformMatchingTex(Line1,Line2))
        self.next_slide()

        self.add(b_def)
        self.play(TransformMatchingTex(Line2,Line3))
        self.next_slide()

        self.add(c_def)
        self.play(TransformMatchingTex(Line3,Line4))
        self.next_slide()

        self.play(Line4.animate.to_edge(LEFT), c_def.animate.to_edge(LEFT), b_def.animate.to_edge(LEFT), a_def.animate.to_edge(LEFT), Equation.animate.to_edge(LEFT))

        self.next_slide()

        CHAINRULE = MathTex(r"\frac{dy}{dx} =",r"\frac{dy}{dc}",r"\frac{dc}{db}",r"\frac{db}{da}",r"\frac{da}{dx}").next_to(Equation, RIGHT * 2)
        self.play(Write(CHAINRULE),runtime=0.3)
        self.next_slide()

        DY_DC = MathTex(r"\frac{dy}{dc} = 1").next_to(Line4, RIGHT).scale(0.5)
        DC_DB = MathTex(r"\frac{dc}{db} = \cos(b)").next_to(c_def, RIGHT).scale(0.5)
        DB_DA = MathTex(r"\frac{db}{da} = -\sin(a)").next_to(b_def, RIGHT).scale(0.5)
        DA_DX = MathTex(r"\frac{da}{dx} = 2x").next_to(a_def, RIGHT).scale(0.5)

        self.play(Write(DY_DC), Write(DC_DB), Write(DB_DA), Write(DA_DX))
        self.next_slide()
        CHAINRULE_COMPLETE = MathTex(r"\frac{dy}{dx} =",r"(1)", r"(\cos(b))", r"(-\sin(a))", r"(2x)").next_to(Equation, RIGHT * 2)

        self.play(TransformMatchingTex(CHAINRULE,CHAINRULE_COMPLETE))
        #endregion
        #SLIDE 4 - AUTODIFF
        self.next_slide()
        self.clear()
        #region Slide4
        SLIDE3_Title = Tex("Automatic Differentiation").to_corner(UP + LEFT)

        self.play(FadeIn(SLIDE3_Title))

        self.next_slide()

        OperationCode = Code(code = """def sin(self):
        result = Scalar(math.sin(self.data), {self,})

        def back():
            self.grad += math.cos(self.data) * result.grad

        result.back = back

        return result""", language="python").next_to(SLIDE3_Title,DOWN).to_edge(LEFT)

        self.play(FadeIn(OperationCode))

        self.next_slide()

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
            s.back())""", language="python").next_to(SLIDE3_Title,DOWN).to_edge(LEFT)

        self.play(FadeOut(OperationCode))
        self.play(FadeIn(BackPropCode))

        #endregion
        #SLIDE 5 - GRADIENT DESCENT
        self.next_slide()
        self.clear()
        #region Slide5
        SLIDE4_Title = Tex("Gradient Descent").to_corner(UP + LEFT)

        self.play(FadeIn(SLIDE4_Title))

        self.next_slide()

        GradientDescentEQN = MathTex(r"a_{n+1} = a_{n} - \alpha\frac{df(a)}{da}").next_to(SLIDE4_Title,DOWN)

        self.play(FadeIn(GradientDescentEQN))
     
        self.next_slide()

        Axes1 = Axes(x_range=[-5,5,1], y_range=[0,25,5], axis_config={"include_numbers":True},x_length=12,y_length=12).scale(0.5).to_edge(RIGHT)

        x2graph = Axes1.plot(lambda x: x ** 2, x_range=[-5,5], use_smoothing=False, color=TEAL_B)

        fx = MathTex(r"f(a) = a^{2}").scale(0.7).next_to(Axes1, UP).shift(LEFT * 0.8)
        fprimex = MathTex(r"f^{\prime}(a) = 2a").scale(0.7).next_to(fx, RIGHT)

        self.add(Axes1)
        self.play(Write(x2graph),Write(fx),Write(fprimex))

        A0 = MathTex(r"a_{0} = 4.5").scale(0.8).next_to(GradientDescentEQN, DOWN)
        Alpha = MathTex(r"\alpha = 0.1").scale(0.8).next_to(A0, DOWN)

        Adot = Dot(color=RED_E,radius=0.1)
        Adot.move_to(Axes1.coords_to_point(4.5, 4.5**2))

        self.add(Adot)
        self.play(FadeIn(A0))
        self.play(FadeIn(Alpha))
        self.next_slide()

        A1L1 = MathTex(r"a_{1} =",r"a_{0}",r"-0.1",r"(2a_{0})").scale(0.8).next_to(Alpha, DOWN)
        A1L2 = MathTex(r"a_{1} = 4.5 - 0.9 = 3.6").scale(0.8).next_to(A1L1, DOWN)

        self.play(Write(A1L1))
        self.play(Write(A1L2))

        self.play(Adot.animate.move_to(Axes1.coords_to_point(3.6, 3.6**2)))

        self.next_slide()

        A2L1 = MathTex(r"a_{2} =",r"a_{1}",r"-0.1",r"(2a_{1})").scale(0.8).next_to(Alpha, DOWN)
        A2L2 = MathTex(r"a_{2} = 3.6 - 0.72 = 2.88").scale(0.8).next_to(A2L1, DOWN)

        self.play(Transform(A1L1,A2L1))
        self.play(Transform(A1L2,A2L2))

        self.play(Adot.animate.move_to(Axes1.coords_to_point(2.88, 2.88**2)))


        self.next_slide()

        A3 = MathTex(r"a_{3} = 2.304").scale(0.8).next_to(A2L2, DOWN)
        A4 = MathTex(r"a_{4} = 1.8432").scale(0.8).next_to(A3, DOWN)
        A5 = MathTex(r"a_{5} = 1.4746").scale(0.8).next_to(A4, DOWN)
        DOT = MathTex(r"\ddots").scale(0.8).next_to(A5, DOWN)
        A20 = MathTex(r"a_{20} = 0.0519").scale(0.8).next_to(DOT, DOWN)
        self.start_loop()
        self.play(FadeIn(A3))
        self.play(Adot.animate.move_to(Axes1.coords_to_point(2.304, 2.304**2)))

        self.play(FadeIn(A4))
        self.play(Adot.animate.move_to(Axes1.coords_to_point(1.8432, 1.8432**2)))

        self.play(FadeIn(A5))
        self.play(Adot.animate.move_to(Axes1.coords_to_point(1.4746, 1.4746**2)))

        self.play(FadeIn(DOT))

        self.play(FadeIn(A20))
        self.play(Adot.animate.move_to(Axes1.coords_to_point(0.0519, 0.0519**2)))
        self.play(FadeOut(A3,A4,A5,DOT,A20))
        self.end_loop()
        #endregion
        #SLIDE 6 - UNIVERSAL FUNCTION APPROXIMATORS
        self.clear()
        #region Slide6
        SLIDE5_Title = Tex("Piecing it all together").to_corner(UP + LEFT)

        self.play(FadeIn(SLIDE5_Title))
        self.next_slide()

        example_function = MathTex(r"f(x_{1},...,x_{n},\text{parameters})").next_to(SLIDE5_Title)

        self.play(FadeIn(example_function))

        weights_mat = MathTex(r"\begin{bmatrix} w_{1,1} & \cdots & w_{1,n} \\ \vdots & \ddots & \vdots \\ w_{m,1} & \cdots & w_{m,n} \end{bmatrix}").to_edge(LEFT)
        inputs_vec = MathTex(r"\begin{bmatrix} x_{1} \\ \vdots \\ x_{n}\end{bmatrix}").next_to(weights_mat, RIGHT)
        bias_vec = MathTex(r" + \begin{bmatrix} b_{1} \\ \vdots \\ b_{m} \end{bmatrix}").next_to(inputs_vec, RIGHT)
        out_vec = MathTex(r"= \begin{bmatrix} a_{1} \\ \vdots \\ a_{m} \end{bmatrix}").next_to(bias_vec, RIGHT)

        Axes2 = Axes(x_range=[-5,5,1], y_range=[-1.25,1.25,0.25], axis_config={"include_numbers":True},x_length=10,y_length=10).scale(0.5).to_edge(RIGHT)
        tanh = Axes2.plot(lambda x: math.tanh(x), x_range=[-5,5], use_smoothing=False, color=TEAL_B)

        tanh_text = MathTex(r"f(x) = \tanh(x)").next_to(Axes2, DOWN)

        self.play(Write(inputs_vec),Write(weights_mat),Write(bias_vec),Write(out_vec))

        self.next_slide()

        self.play(FadeOut(inputs_vec),FadeOut(weights_mat),FadeOut(bias_vec),out_vec.animate.to_edge(LEFT))

        out_vec_AF = MathTex(r"\tanh\begin{bmatrix} a_{1} \\ \vdots \\ a_{m} \end{bmatrix}").to_edge(LEFT)

        out_vec_POST_AF = MathTex(r"= \begin{bmatrix} z_{1} \\ \vdots \\ z_{m} \end{bmatrix}").next_to(out_vec_AF, RIGHT)

        self.play(Write(Axes2))
        self.play(Write(tanh), Write(tanh_text))
        self.next_slide()

        self.play(TransformMatchingTex(out_vec, out_vec_AF))
        self.play(Write(out_vec_POST_AF))

        self.next_slide()
        #endregion
        #SLIDE 7 - IMPROVED FISHSTICK
        self.clear()
        SLIDE5_Title = Tex("Solving equations using this apporach.").to_corner(UP + LEFT)

        self.play(FadeIn(SLIDE5_Title))
        self.next_slide()

        HEAT_EQN = MathTex(r"\frac{du}{dt} =", r"\alpha^{2}(\frac{d^{2}u}{dx^{2}}+\frac{d^{2}u}{dy^{2}})").to_edge(UP).shift(DOWN)

        HEAT_EQN_0 = MathTex(r"\frac{du}{dt} - \frac{d^{2}u}{dx^{2}} - \frac{d^{2}}{dy^{2}}", r"= 0").to_edge(UP).shift(DOWN)

        INITIAL_CONDITIONS = MathTex(r"u(x,y,0) = \frac{1}{2\pi \sigma^{2}}e^{-\frac{(x-x_{0})^{2} + (y-y_{0})^{2}}{2\sigma^{2}}}").next_to(HEAT_EQN_0, DOWN)
        
        BOUNDARY_CONDITIONS1 = MathTex(r"u(0,y,t) = u(x,0,t) = 0").next_to(INITIAL_CONDITIONS, DOWN)
        BOUNDARY_CONDITIONS2 = MathTex(r"u(1,y,t) = u(x,1,t) = 0").next_to(BOUNDARY_CONDITIONS1, DOWN)

        SOLVER_EQN = MathTex(r"f(x,y,t,\text{parameters}) \approx u(x,y,t)").next_to(BOUNDARY_CONDITIONS2,DOWN)

        self.play(FadeIn(HEAT_EQN))
        self.next_slide()
        self.play(Transform(HEAT_EQN,HEAT_EQN_0))
        self.next_slide()
        self.play(FadeIn(INITIAL_CONDITIONS))
        self.next_slide()
        self.play(FadeIn(BOUNDARY_CONDITIONS1), FadeIn(BOUNDARY_CONDITIONS2))
        self.next_slide()
        self.play(FadeIn(SOLVER_EQN))
        self.next_slide()
        #SLIDE 8 - Result
        self.clear()
        SLIDE6_Title = Tex("Results").to_corner(UP + LEFT)

        self.play(FadeIn(SLIDE6_Title))
        


        

