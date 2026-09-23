import React, { useId } from 'react';
import './Dropdown.css';

export interface DropdownOption {
  value: string;
  label: string;
}

export interface DropdownProps {
  label?: string;
  options: DropdownOption[];
  value: string;
  onChange: (value: string) => void;
  ariaLabel?: string;
  disabled?: boolean;
}

export const Dropdown: React.FC<DropdownProps> = ({
  label,
  options,
  value,
  onChange,
  ariaLabel,
  disabled = false,
}) => {
  const selectId = useId();

  return (
    <div className="sg-dropdown-field">
      {label && (
        <label htmlFor={selectId} className="sg-dropdown-label">
          {label}
        </label>
      )}
      <div className="sg-dropdown-wrapper">
        <select
          id={selectId}
          className="sg-dropdown-select"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          aria-label={ariaLabel || label}
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <span className="sg-dropdown-arrow" aria-hidden="true">
          ▼
        </span>
      </div>
    </div>
  );
};
