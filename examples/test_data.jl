using Pkg; Pkg.activate(joinpath(@__DIR__, ".."));
using ReLUQP


function copyToClipboard(matrix; prefix="") # Copies as a comma separated array in column major order
    out = ""
    for element in matrix' # Transpose because eigen expects left to right, top to bottom ordering
        out *= string(element)
        out *= ", "
    end
    clipboard(prefix * out[1:end-2] * ";")
end

H = [6.71383  0.71247  3.75614  0.886201   0.74104;
     0.71247  3.54515  0.835424 0.532941   1.42631;
     3.75614  0.835424 5.22627  0.317399   1.3052;
     0.886201 0.532941 0.317399 4.51379   -0.836364;
     0.74104  1.42631  1.3052   -0.836364  4.98613]

g = [ 2.78913;
    2.34454;
    0.846321;
    4.11682;
    -2.9479]

A = [
    0.84794  0.821944    0.840257 -0.136093   -0.385084;
    -0.203127 -0.0350187 -0.70468   0.239193   -0.105933;
    0.629534 -0.56835    0.762124 -0.437881   -0.547787;
    0.368437  0.900505   0.282161  0.572004   -0.624934;
    -0.447531 -0.166997   0.813608 -0.747849    0.52095;
    0.112888 -0.660786   0.793658 -0.00911187  0.969503
]

l = [-0.73883;
-0.440346;
 0.433656;
-1.47864;
 1.58857;
 0.118917]

u = [ -0.73883;
-0.440346;
0.433656;
-1.47864;
Inf;
Inf]


copyToClipboard(H; prefix="H << ")
copyToClipboard(g; prefix="g << ")
copyToClipboard(A; prefix="A << ")
copyToClipboard(l; prefix="l << ")
copyToClipboard(u; prefix="u << ")


# qp = ReLUQP.QPProb(H, g, A, l, u)

# gpu_qp = ReLUQP.ruiz_equilibration(qp)


m = ReLUQP.setup(H, g, A, l, u; ρ_min=1e-2, ρ_max=1e2);
ReLUQP.CUDA.allowscalar(true)



copyToClipboard(m.gpu_prob.H; prefix="H_ruiz_expected << ")
copyToClipboard(m.gpu_prob.g; prefix="g_ruiz_expected << ")
copyToClipboard(m.gpu_prob.A; prefix="A_ruiz_expected << ")
copyToClipboard(m.gpu_prob.l; prefix="l_ruiz_expected << ")
copyToClipboard(m.gpu_prob.u; prefix="u_ruiz_expected << ")
copyToClipboard(m.gpu_prob.D; prefix="D_ruiz_expected << ")
copyToClipboard(m.gpu_prob.invD; prefix="invD_ruiz_expected << ")
copyToClipboard(m.gpu_prob.E; prefix="E_ruiz_expected << ")
copyToClipboard(m.gpu_prob.invE; prefix="invE_ruiz_expected << ")
copyToClipboard(m.gpu_prob.c; prefix="c_ruiz_expected << ")

bias_mat_rhos_str = ""
for (i, rho_mat) in enumerate(m.bias_mat_ρs)
    bias_mat_rhos_str *= "bias_mat_rhos_expected_" * string(i) * " << "
    for (j, element) in enumerate(rho_mat') # Transpose because eigen expects left to right, top to bottom ordering
        bias_mat_rhos_str *= string(element)
        if j < length(rho_mat)
            bias_mat_rhos_str *= ", "
        else
            bias_mat_rhos_str *= ";\n"
        end
    end
end
clipboard(bias_mat_rhos_str)



######### W #########
layer_rho_Ws_str = ""
for (i, layer) in enumerate(m.layer_ρs)
    layer_rho_Ws_str *= "layer_rhos_expected_W_" * string(i) * " << "
    for (j, element) in enumerate(layer.W') # Transpose because eigen expects left to right, top to bottom ordering
        layer_rho_Ws_str *= string(element)
        if j < length(layer.W)
            layer_rho_Ws_str *= ", "
        else
            layer_rho_Ws_str *= ";\n"
        end
    end
end
clipboard(layer_rho_Ws_str)

######### b #########
layer_rho_bs_str = ""
for (i, layer) in enumerate(m.layer_ρs)
    layer_rho_bs_str *= "layer_rhos_expected_b_" * string(i) * " << "
    for (j, element) in enumerate(layer.b') # Transpose because eigen expects left to right, top to bottom ordering
        layer_rho_bs_str *= string(element)
        println(element)
        if j < length(layer.b)
            layer_rho_bs_str *= ", "
        else
            layer_rho_bs_str *= ";\n"
        end
    end
end
clipboard(layer_rho_bs_str)

######### l #########
layer_rho_ls_str = ""
for (i, layer) in enumerate(m.layer_ρs)
    layer_rho_ls_str *= "layer_rhos_expected_l_" * string(i) * " << "
    for (j, element) in enumerate(layer.l') # Transpose because eigen expects left to right, top to bottom ordering
        layer_rho_ls_str *= string(element)
        println(element)
        if j < length(layer.l)
            layer_rho_ls_str *= ", "
        else
            layer_rho_ls_str *= ";\n"
        end
    end
end
clipboard(layer_rho_ls_str)

######### u #########
layer_rho_us_str = ""
for (i, layer) in enumerate(m.layer_ρs)
    layer_rho_us_str *= "layer_rhos_expected_u_" * string(i) * " << "
    for (j, element) in enumerate(layer.u') # Transpose because eigen expects left to right, top to bottom ordering
        layer_rho_us_str *= string(element)
        println(element)
        if j < length(layer.u)
            layer_rho_us_str *= ", "
        else
            layer_rho_us_str *= ";\n"
        end
    end
end
clipboard(layer_rho_us_str)


##

# Actually solve the problem now
m = ReLUQP.setup(H, g, A, l, u; ρ_min=1e-2, ρ_max=1e2, iters_btw_checks=1);
results = ReLUQP.solve(m)

##

clipboard(results.iters)
clipboard(results.J)
clipboard(m.workspace.primal_res)
clipboard(m.workspace.dual_res)
copyToClipboard(results.x; prefix="x_expected << ")
copyToClipboard(results.z; prefix="z_expected << ")
copyToClipboard(results.λ; prefix="lam_expected << ")