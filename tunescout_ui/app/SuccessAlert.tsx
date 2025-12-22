import { useState, useEffect } from "react";

interface SuccessAlertProps {
  message: string;
  onClose: () => void;
}

export default function SuccessAlert({ message, onClose }: SuccessAlertProps) {
  const [visible, setVisible] = useState(true); // controls opacity
  const [mounted, setMounted] = useState(true); // controls render

  useEffect(() => {
    // Start fade after 3s
    const hideTimer = setTimeout(() => {
      setVisible(false);
      // Remove from DOM 0.5s after fade starts
      const removeTimer = setTimeout(() => {
        setMounted(false);
        onClose?.(); // notify parent if needed
      }, 500);
      return () => clearTimeout(removeTimer);
    }, 1000);

    return () => clearTimeout(hideTimer);
  }, [onClose]);

  if (!mounted) return null;

  return (
    <div
      id="successAlert"
      role="alert"
      className="alert alert-success"
      style={{
        position: "fixed",
        top: "20px",
        zIndex: 9999,
        transition: "opacity 0.5s ease",
        opacity: visible ? 1 : 0,
      }}
    >
      {message}
    </div>
  );
}