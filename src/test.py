def optimize(fe, p_ini, optimizationParams, objectiveHandle, consHandle, numConstraints, generate_rho):
    # 初始化过滤器
    H, Hs = compute_filter_kd_tree(fe)
    ft = {'H': H, 'Hs': Hs}

    p = p_ini  # 控制参数
    loop = 0
    m = numConstraints  # 约束数量
    n = len(p.reshape(-1))  # 控制参数的数量

    # 初始化 MMA 优化器
    mma = MMA()
    mma.setNumConstraints(numConstraints)
    mma.setNumDesignVariables(n)
    mma.setMinandMaxBoundsForDesignVariables(np.zeros((n, 1)), np.ones((n, 1)))

    xval = p.reshape(-1)[:, None]
    xold1, xold2 = xval.copy(), xval.copy()
    mma.registerMMAIter(xval, xold1, xold2)
    mma.setLowerAndUpperAsymptotes(np.ones((n, 1)), np.ones((n, 1)))
    mma.setScalingParams(1.0, np.zeros((m, 1)), 10000 * np.ones((m, 1)), np.zeros((m, 1)))
    mma.setMoveLimit(optimizationParams['movelimit'])

    while loop < optimizationParams['maxIters']:
        loop += 1

        # 通过控制参数生成 rho
        rho = generate_rho(p)

        print(f"MMA solver...")

        # 计算目标函数和约束
        J, dJ = objectiveHandle(rho)
        vc, dvc = consHandle(rho, loop)

        # 应用灵敏度过滤器
        dJ, dvc = applySensitivityFilter(ft, rho, dJ, dvc)

        J, dJ = J, dJ.reshape(-1)[:, None]
        vc, dvc = vc[:, None], dvc.reshape(dvc.shape[0], -1)

        print(f"J.shape = {J.shape}")
        print(f"dJ.shape = {dJ.shape}")
        print(f"vc.shape = {vc.shape}")
        print(f"dvc.shape = {dvc.shape}")

        J, dJ, vc, dvc = np.array(J), np.array(dJ), np.array(vc), np.array(dvc)

        start = time.time()

        # 设置 MMA 的目标和约束
        mma.setObjectiveWithGradient(J, dJ)
        mma.setConstraintWithGradient(vc, dvc)
        mma.mmasub(xval)
        xmma, _, _ = mma.getOptimalValues()

        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()

        mma.registerMMAIter(xval, xold1, xold2)
        p = xval.reshape(p.shape)  # 更新控制参数

        end = time.time()

        time_elapsed = end - start

        print(f"MMA took {time_elapsed} [s]")
        print(f'Iter {loop:d}; J {J:.5f}; constraint {vc}\n\n\n')

    return p
