    private void goodG2B() throws Throwable
    {
        Integer data;

        /* FIX: hardcode data to non-null */
        data = Integer.valueOf(5);

        /* POTENTIAL FLAW: null dereference will occur if data is null */
        IO.writeLine("" + data.toString());

    }
