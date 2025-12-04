/**
 * Authentication utilities
 * Simple session-based auth without external providers
 */

import bcrypt from 'bcryptjs';
import { findUser, findUserById } from './db';
import { User } from './types';

export function authenticateUser(email: string, password: string): User | null {
  const user = findUser(email);
  
  if (!user) return null;

  const isValid = bcrypt.compareSync(password, user.password_hash);
  if (!isValid) return null;

  // Return user without password
  return {
    id: user.id,
    email: user.email,
    role: user.role,
    name: user.name,
    created_at: user.created_at
  };
}

export function getUserById(id: number): User | null {
  const user = findUserById(id);
  
  if (!user) return null;

  // Return user without password
  return {
    id: user.id,
    email: user.email,
    role: user.role,
    name: user.name,
    created_at: user.created_at
  };
}
