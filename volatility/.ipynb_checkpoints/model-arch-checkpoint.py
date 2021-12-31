from arch import arch_model

def arch(returns, p=1):
    model = arch_model(returns, mean='Zero', vol='ARCH', p=p, o=0, q=0)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)