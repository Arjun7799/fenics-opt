from dolfin import *
from fenics_adjoint import Control, ReducedFunctional, IPOPTSolver

# Personal-ID parameters for ID suffix = 7
a, b, c, d, e = 5, 5, 7, 25, 50

def optimize(params):
    nelx, nely = params['nelx'], params['nely']
    volfrac, penal = params['volfrac'], params['penal']
    mu, lam = params['mu'], params['lam']
    load_case = params['case']

    # Mesh, function spaces
    mesh = RectangleMesh(Point(0, 0), Point(100, 30), nelx, nely)
    V_u   = VectorFunctionSpace(mesh, 'CG', 1)
    V_rho = FunctionSpace(mesh, 'CG', 1)

    u   = Function(V_u,   name="Displacement")
    rho = Function(V_rho, name="Density")
    rho.assign(Constant(volfrac))

    def theta(r): return r**penal
    F   = Identity(mesh.topology().dim()) + grad(u)
    psi = lam/2*ln(det(F))**2 + mu*(tr(F.T*F)/2 - mesh.topology().dim()/2 - ln(det(F)))

    dV = dx
    J  = assemble(theta(rho)*psi*dV)

    # BCs
    left   = CompiledSubDomain("near(x[0], 0) && x[1] < a + DOLFIN_EPS", a=a)
    bottom = CompiledSubDomain("near(x[1], 0) && x[0] < b + DOLFIN_EPS", b=b)
    bcs    = [DirichletBC(V_u, Constant((0, 0)), left),
              DirichletBC(V_u, Constant((0, 0)), bottom)]

    # Tractions
    t1 = Constant((0, -10))
    t2 = Constant((0, -7))
    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 100)
    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 30)

    mf = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    Right().mark(mf, 1)
    Top().mark(mf,   2)
    ds = Measure('ds', domain=mesh, subdomain_data=mf)

    if load_case in ('shear', 'combined'):
        J += assemble(dot(t1, u)*ds(1)*theta(rho))
    if load_case in ('top', 'combined'):
        J += assemble(dot(t2, u)*ds(2).where(lambda mf, x: d < x[0] < e)*theta(rho))

    # Solve forward & optimize
    problem   = NonlinearVariationalProblem(derivative(theta(rho)*psi*dV, u), u, bcs)
    solver    = NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver'] = 'newton'
    solver.parameters['linear_solver']    = 'mumps'

    rf         = ReducedFunctional(J, Control(rho))
    vol_constr = lambda r: assemble(r*dV) - volfrac*assemble(Constant(1.0)*dV)
    ipopt      = IPOPTSolver(rf, bounds=(0,1), constraints=[{'type':'eq','fun':vol_constr}])
    rho_opt    = ipopt.solve()

    # Final displacement
    problem_u = NonlinearVariationalProblem(theta(rho_opt)*psi*dV, u, bcs)
    solver_u  = NonlinearVariationalSolver(problem_u)
    solver_u.solve()
    return rho_opt, u.copy(deepcopy=True)
