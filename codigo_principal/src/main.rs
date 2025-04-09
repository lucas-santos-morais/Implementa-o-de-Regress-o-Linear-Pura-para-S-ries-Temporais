fn linear_regression(xs: &[f64], ys: &[f64]) -> Option<(f64, f64)> {
    if xs.len() != ys.len() || xs.is_empty() {
        return None; // Verifica se os dados são válidos
    }

    let n = xs.len() as f64;
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = xs.iter().map(|x| x * x).sum();

    let denominator = n * sum_x2 - sum_x * sum_x;
    if denominator == 0.0 {
        return None; // Evita divisão por zero
    }

    let a = (n * sum_xy - sum_x * sum_y) / denominator;
    let b = (sum_y * sum_x2 - sum_x * sum_xy) / denominator;

    Some((a, b)) // Retorna os coeficientes da equação da reta
}

fn calcular_r2(xs: &[f64], ys: &[f64], a: f64, b: f64) -> f64 {
    let media_y = ys.iter().sum::<f64>() / ys.len() as f64;
    let mut ss_total = 0.0;
    let mut ss_residual = 0.0;

    for i in 0..ys.len() {
        let previsao = a * xs[i] + b;
        ss_total += (ys[i] - media_y).powi(2);
        ss_residual += (ys[i] - previsao).powi(2);
    }

    1.0 - (ss_residual / ss_total)
}

fn calcular_mse(xs: &[f64], ys: &[f64], a: f64, b: f64) -> f64 {
    let mut erro_total = 0.0;

    for i in 0..ys.len() {
        let previsao = a * xs[i] + b;
        erro_total += (ys[i] - previsao).powi(2);
    }

    erro_total / ys.len() as f64
}

fn predict(x_future: f64, a: f64, b: f64) -> f64 {
    a * x_future + b
}

fn main() {
    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Índices de tempo
    let ys = vec![2.0, 4.1, 6.0, 8.2, 10.1]; // Valores da série temporal

    if let Some((a, b)) = linear_regression(&xs, &ys) {
        println!("Coeficiente angular (a): {:.4}", a);
        println!("Coeficiente linear (b): {:.4}", b);

        // Cálculo das métricas de avaliação
        let r2 = calcular_r2(&xs, &ys, a, b);
        let mse = calcular_mse(&xs, &ys, a, b);

        println!("Coeficiente de Determinação (R²): {:.4}", r2);
        println!("Erro Quadrático Médio (MSE): {:.4}", mse);

        // Previsão
        let x_future = 6.0; // Previsão para o próximo período
        let y_future = predict(x_future, a, b);
        println!("Previsão para x = {}: {:.4}", x_future, y_future);
    } else {
        println!("Erro ao calcular regressão linear.");
    }
}