CREATE TABLE `sp500_performance` (
  `performance_id` BIGINT(20) NOT NULL AUTO_INCREMENT,
  `report_date` DATE NOT NULL,
  `rounded_start_date` DATE NOT NULL,
  `rounded_end_date` DATE NOT NULL,
  `start_value` INT(32) DEFAULT NULL,
  `end_value` INT(32) DEFAULT NULL,
  `performance_over_range` DECIMAL(12,2) DEFAULT NULL,
  PRIMARY KEY (`performance_id`)
)
COLLATE='latin1_swedish_ci'
ENGINE=InnoDB
;