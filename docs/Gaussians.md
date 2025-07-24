
### Gaussian ellipsoid constant

The constant, $C$ to apply to the result of the Gaussian integral to normalize it to the 
equivalent ellipsoid volume.

*Quadratic Equation for ellipsoid*

$$ (x-u)^T\mathbf{A}(x-u)=1 $$

Where

$$ \mathbf{A = UDV^T = UDU^T }$$

$\mathbf{D}$ is diagonal matrix with the eigenvalues 
$d_{00}d_{11}d_{22} = \frac{1}{a^2b^2c^2}$ where $abc$ are the magnitudes 
of the ellipsoid axes and the eigenvectors in $\mathbf{U}$ are the unit vectors 
on the ellipsoid vectors.

$$ \mathbf{|A| = |UDV^T| = |U||D||U^T| = |D|} = \frac{1}{a^2b^2c^2}$$

*The Gaussian representation for ellipsoid*

$$
\begin{align*}
f(x) &= e^{-n(x-u)^T\mathbf{A}(x-u)} \\
\int_x f(x) dx &= \sqrt{\frac{\pi^3}{n^3\mathbf{|A|}}} = abc\sqrt{\frac{\pi^3}{n^3}} 
\end{align*}
$$
*Volume constant*
$$
\begin{align*}
C\int_x f(x) dx &= \frac{4}{3}\pi abc \\
C &= \frac{4{\pi}n^\frac{3}{2}}{3\pi^\frac{3}{2}} = \frac{4n^\frac{3}{2}}{3\sqrt{(\pi)}} = 0.75225 n^\frac{3}{2}
\end{align*}
$$

### Product of 2 ellipsoid Gaussians

Adapted from https://arxiv.org/pdf/1811.04751v1.  

For symmetric matrix $A$, $A^T = A$ and 

$$x^TAu = (x^TAu)^T = u^TA^Tx = u^TAx $$


One gaussian at the origin

$$
\begin{align*}
f(x) = & Ce^{-n(x-u)^TA(x-u)}Ce^{-nx^TBx} \\
=& C^2 e^{-n(x-u)^TA(x-u)-nx^TBx} \\ 
=& C^2 e^{-n\alpha} \\
\alpha =& (x-u)^TA(x-u)+x^TBx \\
=& x^TAx - x^TAu -u^TAx + u^TAu + x^TBx \\
=& x^TAx + x^TBx -x^TAu -x^TA^Tu + u^TAu \\
=& x^T(A + B) - 2x^TAu + u^TAu \\
P =& A + B \\
v =& P^{-1}Au \\
u =& A^{-1}Pv \\
\alpha =& x^TPx -2xTAA^{-1}Pv + u^TAu \\
=& x^TPx -2x^TPv + u^TAu \\
=& x^TPx -x^TPv -v^TPx + v^TPv - v^TPv + u^TAu \\
=& (x-v)^TP(x-v) -v^TPv + u^TAu \\
f(x) =& C^2e^{n({v^TPv - u^TAu})}e^{-n(x-v)^TP(x-v)} \\
\int_x f(x) dx =& \sqrt{\frac{\pi^3}{n^3|P|}}C^2e^{n({v^TPv - u^TAu})}
\end{align*}
$$

The original paper substitutes $A$ and $B$ and $u$ back into the integral

### Integral of ellipsoid Gaussian with itself

$$
\begin{align*}
\int_x f(x) dx =& \sqrt{\frac{\pi^3}{n^3|P|}}C^2e^{n({v^TPv - u^TAu})}
A =& B \\
u =& O \\
v =& O, P = 2A, |P| = 8|A| \\
\int_x f(x) dx =& \sqrt{\frac{\pi^3}{n^3|P|}}C^2 \\
=& \sqrt{\frac{\pi^3}{n^38|A|}} . [\frac{4n^\frac{3}{2}}{3\sqrt{\pi}}]^2 \\
=& \frac{4}{9}abc\sqrt{2{\pi}n^3} \\
=& \frac{4}{3}{\pi}abcn^\frac{3}{2}\sqrt{\frac{2}{9\pi}} \\
=& volume \times 0.2660 \times n^\frac{3}{2}
\end{align*}
$$
 
Or the integral of a Gaussian ellipsoid with itself is 3.760 times smaller than the volume when n is 1. 
If n is 2.418 then the Gaussian volume with itself is the same as the ellipsoid volume

### Ellipsoid rigid body optimization


I don't believe that we can use the Rigid body equations for atom centered potentials 
for the asymmetric Gaussians that we are going to use to approximate ellipsoids.

On the first page the volume is calculated from the summation of atomic volumes where 
V the atomic volume and 
$R_i(q)$ returns the rotated atom coordinates.

*For a potential that depends only on atom centers:*

$$ 
\begin{align*}
 \overline{V}(q) &= \sum_i V(\mathbf{R_i}(q))  \\
&= \sum_i V(\mathbf{R}x_i^0) \\
x_i &= \mathbf{R}x_i^0 \\
\end{align*}
$$

*For a potential that depends on ellipsoids at a center:*


$$ (x-x_{i}^0)^T\mathbf{A_i^0}(x-x_i^0)  = 1 $$

An ellipsoid (at the origin) rotated by $\mathbf{R}$ (where $\mathbf{R^{-1} = R^T}$)

$$ 
\begin{align*}
x' &= \mathbf{R}x \\
x^T \mathbf(A_i^0) x &= 1 \\
(\mathbf{R^{-1}}x')^T\mathbf{A_i^0}(\mathbf{R^{-1}}x')  &= 1 \\
x'^T\mathbf{R}\mathbf{A_{i}^0}\mathbf{R^T}x' &= 1 \\
\mathbf{A_i} &= \mathbf{RA_i^0R^T}
\end{align*}
$$

$$
 \overline{V}(q) =  \sum V(\mathbf{Rx_i^0}, \mathbf{R^TA_i^0R})
$$


### Covariance Gaussian

Normalized covariance Gaussian
$$
\begin{align*}
f(x) =& \frac {1}{\sqrt(8\pi^3|\Sigma|)}e^{-\frac{1}{2}}(x-u)^T\Sigma^{-1}(x-u)
\end{align*}
$$

Unnormalized

$$
\begin{align*}
f(x) =& e^{-\frac{1}{2}}(x-u)^T\Sigma^{-1}(x-u) \\
\int_x f(x) dx =& \sqrt{\frac {8\pi^3} {|\Sigma^{-1}|}} \\
\frac {1}{2} \Sigma^{-1} =& \Alpha &\\
\Sigma^{-1} =& 2\Alpha  \\
| \Sigma^{-1} | =& 8 |\Alpha | \\


\end{align*}
$$