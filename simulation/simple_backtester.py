"""
사전 계산된 시그널과 비용 모델을 바탕으로 간단한 이벤트 기반 백테스팅을 수행하여 전략의 성과를 검증합니다.

이 파일의 목적:
- 정의된 시뮬레이션 규칙(MVP 기준)에 따라 시간 순서대로 거래를 실행하여 재현 가능한 결과를 도출합니다.
- 최종 결과물로 자산 추이(Equity Curve)와 상세 거래 로그(Trade Log)를 생성하여 전략을 평가할 기초 데이터를 제공합니다.

사용되는 변수와 함수 목록:
- 변수
  - equity: 현재 총자산(현금 + 평가금액) / 단위: 원
  - cash: 보유 현금 / 단위: 원
  - position_qty: 보유 포지션 수량 / 단위: 개
  - position_avg_price: 보유 포지션의 평균 진입 단가 / 단위: 원
- 함수
  - calculate_position_size()
    - 역할: Fixed Fractional 위험 관리 모델에 따라 진입할 포지션 수량을 계산
    - 입력값:
        - equity: float - 현재 총자산
        - risk_per_trade: float - 거래당 허용할 위험 비율 (예: 0.02)
        - entry_price: float - 예상 진입 가격
        - stop_price: float - 손절 가격
    - 반환값: float - 내림(floor) 처리된 최종 주문 수량

  - run_simulation()
    - 역할: 메인 백테스팅 루프를 실행하여 거래를 시뮬레이션하고 결과를 기록
    - 입력값:
        - data: pandas.DataFrame - 'open', 'low', 'close', 'signal', 'stop_threshold' 컬럼을 포함한 시계열 데이터
        - initial_capital: float = 1,000,000 - 초기 자본금 (원)
        - risk_per_trade: float = 0.02 - 거래당 위험 비율
        - commission_rate: float = 0.0005 - 거래 수수료 비율 (예: 0.05%)
        - slippage_rate: float = 0.0005 - 슬리피지 비율 (예: 0.05%)
    - 반환값: (pandas.DataFrame, pandas.DataFrame) - (자산 추이 데이터프레임, 거래 로그 데이터프레임)

파일의 흐름(→ / ->):
- 시뮬레이션 초기화 (자본금, 포지션 상태 설정) -> 입력 데이터 각 행(Bar) 순회 시작
-> (포지션 보유 시) 청산 조건 확인 (스탑 우선) -> (포지션 미보유 시) 진입 신호 확인
-> (진입 시) 포지션 크기 계산 -> 주문 생성(진입/청산) -> 다음 봉 시가 기준으로 체결 처리 (수수료, 슬리피지 반영)
-> 계좌 상태 갱신 (현금, 포지션, 자본) -> 현재 자산 가치 기록 -> 루프 종료 후 최종 결과 반환
"""
import pandas as pd
import numpy as np

def calculate_position_size(equity, risk_per_trade, entry_price, stop_price):
    if entry_price <= stop_price:
        return 0
    
    risk_amount = equity * risk_per_trade
    risk_per_unit = entry_price - stop_price
    
    if risk_per_unit <= 0:
        return 0
        
    quantity = risk_amount / risk_per_unit
    return np.floor(quantity)

def run_simulation(data, initial_capital=1_000_000, risk_per_trade=0.02, commission_rate=0.0005, slippage_rate=0.0005):
    
    equity = initial_capital
    cash = initial_capital
    position_qty = 0
    position_avg_price = 0
    
    trade_log = []
    equity_curve = []

    required_cols = ['open', 'low', 'close', 'signal', 'stop_threshold']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Input data must contain columns: {required_cols}")

    for timestamp, row in data.iterrows():
        current_open = row['open']
        current_low = row['low']
        current_close = row['close']
        signal = row['signal']
        stop_threshold = row['stop_threshold']

        exit_reason = None
        
        if position_qty > 0:
            is_stop_hit = current_low <= stop_threshold if pd.notna(stop_threshold) else False
            is_signal_exit = signal == 0
            
            if is_stop_hit:
                exit_reason = 'stop_loss'
            elif is_signal_exit:
                exit_reason = 'signal'

        if exit_reason:
            sell_price = current_open * (1 - slippage_rate)
            trade_value = sell_price * position_qty
            commission = trade_value * commission_rate
            
            pnl = (sell_price - position_avg_price) * position_qty - commission
            
            cash += trade_value - commission
            equity = cash
            
            trade_log.append({
                'exit_time': timestamp,
                'exit_price': sell_price,
                'pnl': pnl,
                'reason': exit_reason,
                'qty': position_qty,
                'entry_time': entry_timestamp,
                'entry_price': position_avg_price,
            })
            
            position_qty = 0
            position_avg_price = 0

        elif position_qty == 0 and signal == 1:
            size = calculate_position_size(equity, risk_per_trade, current_open, stop_threshold)
            
            if size > 0:
                buy_price = current_open * (1 + slippage_rate)
                trade_value = buy_price * size
                commission = trade_value * commission_rate

                if cash >= trade_value + commission:
                    cash -= (trade_value + commission)
                    position_qty = size
                    position_avg_price = buy_price
                    entry_timestamp = timestamp
                else:
                    trade_log.append({
                        'entry_time': timestamp,
                        'reason': 'insufficient_funds',
                        'pnl': 0,
                    })

        portfolio_value = cash + (position_qty * current_close)
        equity_curve.append({'timestamp': timestamp, 'equity': portfolio_value})

    return pd.DataFrame(equity_curve).set_index('timestamp'), pd.DataFrame(trade_log)

if __name__ == '__main__':
    
    sample_dates = pd.to_datetime(pd.date_range(start='2024-01-01', periods=100, freq='D'))
    
    sample_data = pd.DataFrame(index=sample_dates)
    sample_data['open'] = np.random.uniform(95, 105, size=100).cumsum() + 1000
    sample_data['low'] = sample_data['open'] - np.random.uniform(0, 5, size=100)
    sample_data['close'] = sample_data['open'] + np.random.uniform(-2, 2, size=100)
    
    sample_data['signal'] = np.random.choice([0, 1], size=100, p=[0.8, 0.2])
    sample_data['signal'] = pd.Series(sample_data['signal']).rolling(window=5).max().fillna(0).astype(int)
    
    sample_data['stop_threshold'] = sample_data['low'].shift(1).rolling(window=10).min()
    
    print("--- Sample Input Data ---")
    print(sample_data.head())
    print("-" * 25)
    
    equity_df, trades_df = run_simulation(sample_data)
    
    print("\n--- Equity Curve (First 5 Rows) ---")
    print(equity_df.head())
    print("-" * 25)

    print("\n--- Trade Log (First 5 Rows) ---")
    if not trades_df.empty:
        print(trades_df.head())
    else:
        print("No trades were executed.")
    print("-" * 25)

    equity_df.to_csv('equity_curve.csv')
    trades_df.to_csv('trade_log.csv')
    
    print("\n[SUCCESS] Results saved to 'equity_curve.csv' and 'trade_log.csv'")