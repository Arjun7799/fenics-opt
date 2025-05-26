import matplotlib.pyplot as plt
from dolfin import plot
from ipopt_neohooke_optimize import optimize

nelx, nely = 100, 30
volfrac    = 0.5
penal      = 3
E, nu      = 10.0, 0.45
mu  = E / (2*(1+nu))
lam = E*nu/((1+nu)*(1-2*nu))

cases = {
  'top':      'Top Load (t2)',
  'shear':    'Shear Load (t1)',
  'combined': 'Combined Load'
}

for key,title in cases.items():
    params = dict(nelx=nelx, nely=nely, volfrac=volfrac,
                  penal=penal, mu=mu, lam=lam, case=key)
    rho_opt, _ = optimize(params)

    plt.figure(figsize=(6,2))
    p = plot(rho_opt)
    plt.title(title)
    plt.axis('off')
    plt.colorbar(p)
    plt.tight_layout()
    fname = f'{key}_density.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')
