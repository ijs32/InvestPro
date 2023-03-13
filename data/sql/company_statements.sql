CREATE TABLE `predicted_categories` (
  `statement_id` BIGINT(20) NOT NULL AUTO_INCREMENT, --
  `ticker` VARCHAR(10) NOT NULL, -- 
  `fiscalDateEnding` DATE NOT NULL, -- 
  `reportedCurrency` VARCHAR(10) NOT NULL, -- 
  `totalAssets` DECIMAL(20,2) NOT NULL, -- 
  `totalCurrentAssets` DECIMAL(20,2) NOT NULL, --
  `cashAndCashEquivalentsAtCarryingValue` DECIMAL(20,2) NOT NULL, --
  `cashAndShortTermInvestments` DECIMAL(20,2) NOT NULL, --
  `inventory` DECIMAL(20,2) NOT NULL, --
  `currentNetReceivables` DECIMAL(20,2) NOT NULL, --
  `totalNonCurrentAssets` DECIMAL(20,2) NOT NULL, -- 
  `propertyPlantEquipment` DECIMAL(20,2) NOT NULL, --
  `accumulatedDepreciationAmortizationPPE` DECIMAL(20,2) NOT NULL, --
  `intangibleAssets` DECIMAL(20,2) NOT NULL, --
  `intangibleAssetsExcludingGoodwill` DECIMAL(20,2) NOT NULL, --
  `goodwill` DECIMAL(20,2) NOT NULL, --
  `investments` DECIMAL(20,2) NOT NULL, --
  `longTermInvestments` DECIMAL(20,2) NOT NULL, --
  `shortTermInvestments` DECIMAL(20,2) NOT NULL, --
  `otherCurrentAssets` DECIMAL(20,2) NOT NULL, --
  `otherNonCurrentAssets` DECIMAL(20,2) NOT NULL, --
  `totalLiabilities` DECIMAL(20,2) NOT NULL, -- 
  `totalCurrentLiabilities` DECIMAL(20,2) NOT NULL, --
  `currentAccountsPayable` DECIMAL(20,2) NOT NULL, --
  `deferredRevenue` DECIMAL(20,2) NOT NULL, --
  `currentDebt` DECIMAL(20,2) NOT NULL,
  `shortTermDebt` DECIMAL(20,2) NOT NULL,
  `totalNonCurrentLiabilities` DECIMAL(20,2) NOT NULL,
  `capitalLeaseObligations` DECIMAL(20,2) NOT NULL,
  `longTermDebt` DECIMAL(20,2) NOT NULL,
  `currentLongTermDebt` DECIMAL(20,2) NOT NULL,
  `longTermDebtNoncurrent` DECIMAL(20,2) NOT NULL, --
  `shortLongTermDebtTotal` DECIMAL(20,2) NOT NULL,
  `otherCurrentLiabilities` DECIMAL(20,2) NOT NULL,
  `otherNonCurrentLiabilities` DECIMAL(20,2) NOT NULL,
  `totalShareholderEquity` DECIMAL(20,2) NOT NULL,
  `treasuryStock` DECIMAL(20,2) NOT NULL,
  `retainedEarnings` DECIMAL(20,2) NOT NULL,
  `commonStock` DECIMAL(20,2) NOT NULL,
  `commonStockSharesOutstanding` DECIMAL(20,2) NOT NULL, --
  `operatingCashflow` DECIMAL(20,2) NOT NULL,
  `paymentsForOperatingActivities` DECIMAL(20,2) NOT NULL,
  `proceedsFromOperatingActivities` DECIMAL(20,2) NOT NULL,
  `changeInOperatingLiabilities` DECIMAL(20,2) NOT NULL, --
  `changeInOperatingAssets` DECIMAL(20,2) NOT NULL,
  `depreciationDepletionAndAmortization` DECIMAL(20,2) NOT NULL,
  `capitalExpenditures` DECIMAL(20,2) NOT NULL,
  `changeInReceivables` DECIMAL(20,2) NOT NULL,
  `changeInInventory` DECIMAL(20,2) NOT NULL,
  `profitLoss` DECIMAL(20,2) NOT NULL, --
  `cashflowFromInvestment` DECIMAL(20,2) NOT NULL,
  `cashflowFromFinancing` DECIMAL(20,2) NOT NULL,
  `proceedsFromRepaymentsOfShortTermDebt` DECIMAL(20,2) NOT NULL,
  `paymentsForRepurchaseOfCommonStock` DECIMAL(20,2) NOT NULL,
  `paymentsForRepurchaseOfEquity` DECIMAL(20,2) NOT NULL, --
  `paymentsForRepurchaseOfPreferredStock` DECIMAL(20,2) NOT NULL,
  `dividendPayout` DECIMAL(20,2) NOT NULL,
  `dividendPayoutCommonStock` DECIMAL(20,2) NOT NULL,
  `dividendPayoutPreferredStock` DECIMAL(20,2) NOT NULL,
  `proceedsFromIssuanceOfCommonStock` DECIMAL(20,2) NOT NULL,
  `proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet` DECIMAL(20,2) NOT NULL, --
  `proceedsFromIssuanceOfPreferredStock` DECIMAL(20,2) NOT NULL,
  `proceedsFromRepurchaseOfEquity` DECIMAL(20,2) NOT NULL,
  `proceedsFromSaleOfTreasuryStock` DECIMAL(20,2) NOT NULL,
  `changeInCashAndCashEquivalents` DECIMAL(20,2) NOT NULL, --
  `changeInExchangeRate` DECIMAL(20,2) NOT NULL,
  `netIncome` DECIMAL(20,2) NOT NULL,
  `grossProfit` DECIMAL(20,2) NOT NULL,
  `totalRevenue` DECIMAL(20,2) NOT NULL,
  `costOfRevenue` DECIMAL(20,2) NOT NULL,
  `costofGoodsAndServicesSold` DECIMAL(20,2) NOT NULL,
  `operatingIncome` DECIMAL(20,2) NOT NULL,
  `sellingGeneralAndAdministrative` DECIMAL(20,2) NOT NULL,
  `researchAndDevelopment` DECIMAL(20,2) NOT NULL,
  `operatingExpenses` DECIMAL(20,2) NOT NULL, --
  `investmentIncomeNet` DECIMAL(20,2) NOT NULL,
  `netInterestIncome` DECIMAL(20,2) NOT NULL,
  `interestIncome` DECIMAL(20,2) NOT NULL,
  `interestExpense` DECIMAL(20,2) NOT NULL,
  `nonInterestIncome` DECIMAL(20,2) NOT NULL,
  `otherNonOperatingIncome` DECIMAL(20,2) NOT NULL, --
  `depreciation` DECIMAL(20,2) NOT NULL,
  `depreciationAndAmortization` DECIMAL(20,2) NOT NULL,
  `incomeBeforeTax` DECIMAL(20,2) NOT NULL,
  `incomeTaxExpense` DECIMAL(20,2) NOT NULL,
  `interestAndDebtExpense` DECIMAL(20,2) NOT NULL,
  `netIncomeFromContinuingOperations` DECIMAL(20,2) NOT NULL,
  `comprehensiveIncomeNetOfTax` DECIMAL(20,2) NOT NULL,
  `ebit` DECIMAL(20,2) NOT NULL,
  `ebitda` DECIMAL(20,2) NOT NULL,
  PRIMARY KEY (`prediction_id`)
)
COLLATE='latin1_swedish_ci'
ENGINE=InnoDB
;